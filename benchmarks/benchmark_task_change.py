"""Deterministic offline benchmark for session task-change intent routing.

This benchmark does not call the router HTTP service or any model provider.
It compares:

1. ``current_rule``: the existing keyword/Jaccard implementation.
2. ``semantic_v1``: the same explicit rules plus local BGE similarity against
   the rolling task goal and recent user history.

The dataset is intentionally small and curated. It is a regression baseline,
not a claim of production accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from squilla_api_router.routing_state import (  # noqa: E402
    INTENTS,
    RequestIdentity,
    RouteState,
    _CONTINUE_RE,
    _NEW_TASK_RE,
    _REDO_RE,
    decide_intent,
    text_similarity,
)
from squilla_api_router.v4_runtime.bge_onnx import OnnxBGE  # noqa: E402


DATASET = Path(__file__).with_name("task_change_cases.jsonl")
DEFAULT_OUTPUT = Path(__file__).with_name("task_change_benchmark_latest.json")
_TASK_SHIFT_CUE_RE = re.compile(
    r"(另外|顺便|现在|接下来|换(?:个|成|一个)?|改为|重新做(?:一个)?|"
    r"帮我(?:写|做|分析|设计|实现|比较|生成|翻译|整理|安排)|"
    r"new task|another task|instead|switch to|help me (?:write|build|analy[sz]e|"
    r"design|compare|generate|translate|plan))",
    re.IGNORECASE,
)


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        case = json.loads(raw)
        missing = {
            "id",
            "category",
            "anchor",
            "latest",
            "expected",
        } - set(case)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        if case["id"] in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {case['id']}")
        if case["expected"] not in INTENTS:
            raise ValueError(
                f"{path}:{line_number}: invalid expected intent {case['expected']}"
            )
        seen.add(case["id"])
        cases.append(case)
    return cases


def _identity(case: dict[str, Any]) -> RequestIdentity:
    system_hash = "system-v2" if case.get("system_changed") else "system-v1"
    tools_hash = "tools-v2" if case.get("tools_changed") else "tools-v1"
    latest = str(case["latest"])
    return RequestIdentity(
        session_id=f"benchmark:{case['id']}",
        explicit_session=True,
        fingerprint=f"fingerprint:{case['id']}",
        explicit_intent=str(case.get("explicit_intent") or ""),
        task_id_hint=str(case.get("task_id_hint") or ""),
        system_hash=system_hash,
        tools_hash=tools_hash,
        anchor=str(case.get("task_goal") or case["anchor"]),
        latest_user_text=latest,
        last_role=str(case.get("last_role") or "user"),
    )


def _previous(case: dict[str, Any]) -> RouteState:
    return RouteState(
        session_id=f"benchmark:{case['id']}",
        task_id=str(case.get("previous_task_id") or f"task:{case['id']}"),
        task_anchor=str(case["anchor"]),
        system_hash="system-v1",
        tools_hash="tools-v1",
        tier="c1",
        route_class="R1",
        model="benchmark-model",
        provider="benchmark",
        api_format="openai_chat",
        base_url="http://offline.invalid",
        thinking_mode="T1",
        prompt_policy="P1",
        forced_intent=str(case.get("forced_intent") or ""),
        forced_reason=str(case.get("forced_reason") or ""),
        last_tool_signature="",
        repeated_tool_count=0,
        revision=1,
        turn_count=int(case.get("turn_count") or 1),
        updated_at=0.0,
    )


def _references(case: dict[str, Any]) -> list[str]:
    references = [str(case.get("task_goal") or case["anchor"])]
    history = case.get("history_user_texts") or []
    if isinstance(history, list):
        references.extend(str(item) for item in history[-4:] if str(item).strip())
    return references


def _semantic_similarities(
    encoder: OnnxBGE,
    case: dict[str, Any],
) -> tuple[float, float]:
    latest = str(case["latest"]).strip()
    references = _references(case)
    if not latest or not references:
        return 0.0, 0.0
    vectors = encoder.encode(
        [latest, *references, str(case["anchor"])],
        batch_size=len(references) + 2,
        convert_to_numpy=True,
    )
    current = vectors[0]
    goal_similarities = vectors[1 : len(references) + 1] @ current
    anchor_similarity = float(vectors[-1] @ current)
    return float(np.max(goal_similarities)), anchor_similarity


def _semantic_decision(
    case: dict[str, Any],
    similarity: float,
    *,
    low_threshold: float,
    high_threshold: float,
) -> tuple[str, str]:
    explicit = str(case.get("explicit_intent") or "").strip().lower().replace("-", "_")
    aliases = {
        "new": "new_task",
        "continue": "continue",
        "redo": "redo",
        "regenerate": "redo",
        "task_changed": "task_changed",
    }
    explicit = aliases.get(explicit, explicit)
    if explicit in INTENTS:
        return explicit, "explicit_control"
    if case.get("forced_intent") in INTENTS:
        return str(case["forced_intent"]), "forced_intent"

    previous = _previous(case)
    if case.get("task_id_hint") and case["task_id_hint"] != previous.task_id:
        return "task_changed", "explicit_task_id_changed"

    text = str(case["latest"]).strip()
    if case.get("system_changed"):
        return "task_changed", "system_prompt_changed"
    if case.get("tools_changed"):
        return "task_changed", "tool_schema_changed"
    if _NEW_TASK_RE.search(text):
        return "task_changed", "explicit_new_task_phrase"
    if _REDO_RE.search(text):
        return "redo", "explicit_redo_phrase"
    if _CONTINUE_RE.search(text):
        return "continue", "continuation_phrase"
    if str(case.get("last_role") or "") in {"tool", "function", "tool_result"}:
        return "continue", "tool_result_continuation"
    if text and text in {
        previous.task_anchor.strip(),
        str(case.get("task_goal") or "").strip(),
    }:
        return "redo", "same_user_message_repeated"

    if similarity >= high_threshold:
        return "continue", "semantic_high_similarity"
    if similarity <= low_threshold:
        return "task_changed", "semantic_low_similarity"
    if len(text) >= 12 and _TASK_SHIFT_CUE_RE.search(text):
        return "task_changed", "semantic_mid_with_shift_cue"
    return "continue", "semantic_mid_conservative"


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(INTENTS)
    scores: list[float] = []
    for label in labels:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / len(scores)


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    expected = [str(row["expected"]) for row in rows]
    predicted = [str(row[key]) for row in rows]
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, prediction in zip(expected, predicted):
        confusion[truth][prediction] += 1
    false_split = sum(
        1
        for truth, prediction in zip(expected, predicted)
        if truth in {"continue", "redo"} and prediction in {"task_changed", "new_task"}
    )
    missed_change = sum(
        1
        for truth, prediction in zip(expected, predicted)
        if truth == "task_changed" and prediction in {"continue", "redo"}
    )
    same_task = [
        float(row["similarity"])
        for row in rows
        if row["expected"] in {"continue", "redo"}
    ]
    changed = [
        float(row["similarity"])
        for row in rows
        if row["expected"] == "task_changed"
    ]
    same_anchor = [
        float(row["anchor_similarity"])
        for row in rows
        if row["expected"] in {"continue", "redo"}
    ]
    changed_anchor = [
        float(row["anchor_similarity"])
        for row in rows
        if row["expected"] == "task_changed"
    ]
    return {
        "cases": len(rows),
        "accuracy": round(sum(e == p for e, p in zip(expected, predicted)) / len(rows), 6),
        "macro_f1": round(_macro_f1(expected, predicted), 6),
        "false_split_count": false_split,
        "false_split_rate": round(
            false_split / max(1, len(same_task)),
            6,
        ),
        "missed_change_count": missed_change,
        "missed_change_rate": round(missed_change / max(1, len(changed)), 6),
        "confusion": {
            truth: dict(sorted(counts.items()))
            for truth, counts in sorted(confusion.items())
        },
        "semantic_similarity": {
            "same_task_median": (
                round(statistics.median(same_task), 6) if same_task else None
            ),
            "changed_task_median": (
                round(statistics.median(changed), 6) if changed else None
            ),
            "same_task_anchor_median": (
                round(statistics.median(same_anchor), 6) if same_anchor else None
            ),
            "changed_task_anchor_median": (
                round(statistics.median(changed_anchor), 6)
                if changed_anchor
                else None
            ),
        },
    }


def _category_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    return {
        category: {
            "cases": len(items),
            "accuracy": round(
                sum(item["expected"] == item[key] for item in items) / len(items),
                6,
            ),
        }
        for category, items in sorted(by_category.items())
    }


def threshold_sweep(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report in-sample threshold sensitivity for the semantic candidate."""
    case_by_id = {case["id"]: case for case in cases}
    candidates: list[dict[str, Any]] = []
    for low_step in range(12, 29):
        low = low_step * 0.025
        for high_step in range(22, 39):
            high = high_step * 0.025
            if high <= low:
                continue
            evaluated: list[dict[str, Any]] = []
            for row in rows:
                prediction, _ = _semantic_decision(
                    case_by_id[row["id"]],
                    float(row["goal_similarity"]),
                    low_threshold=low,
                    high_threshold=high,
                )
                evaluated.append({**row, "sweep_prediction": prediction})
            metrics = _metrics(evaluated, "sweep_prediction")
            candidates.append(
                {
                    "low_threshold": round(low, 3),
                    "high_threshold": round(high, 3),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "false_split_rate": metrics["false_split_rate"],
                    "missed_change_rate": metrics["missed_change_rate"],
                }
            )
    candidates.sort(
        key=lambda item: (
            -item["accuracy"],
            -item["macro_f1"],
            item["false_split_rate"],
            item["missed_change_rate"],
        )
    )
    return {
        "note": "In-sample diagnostic only; do not use the winner as a production threshold.",
        "evaluated_combinations": len(candidates),
        "top": candidates[:10],
    }


def run_benchmark(
    *,
    cases_path: Path = DATASET,
    bge_model_dir: Path = ROOT / "model_bundle" / "bge_onnx",
    low_threshold: float = 0.50,
    high_threshold: float = 0.78,
    rule_similarity_threshold: float = 0.15,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    encoder = OnnxBGE(bge_model_dir)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for case in cases:
        identity = _identity(case)
        previous = _previous(case)
        baseline = decide_intent(
            identity,
            previous,
            explicit_intent=identity.explicit_intent,
            task_similarity_threshold=rule_similarity_threshold,
        )
        goal_similarity, anchor_similarity = _semantic_similarities(encoder, case)
        goal_lexical_similarity = text_similarity(
            str(case.get("task_goal") or case["anchor"]),
            str(case["latest"]),
        )
        semantic, semantic_reason = _semantic_decision(
            case,
            goal_similarity,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "anchor": case.get("task_goal") or case["anchor"],
                "latest": case["latest"],
                "expected": case["expected"],
                "current_rule": baseline.intent,
                "current_rule_reason": baseline.reason,
                "semantic_v1": semantic,
                "semantic_reason": semantic_reason,
                "goal_similarity": round(goal_similarity, 6),
                "anchor_similarity": round(anchor_similarity, 6),
                "similarity": round(goal_similarity, 6),
                "goal_lexical_similarity": round(goal_lexical_similarity, 6),
            }
        )
    elapsed = time.perf_counter() - started
    return {
        "dataset": str(cases_path),
        "config": {
            "bge_model_dir": str(bge_model_dir),
            "semantic_low_threshold": low_threshold,
            "semantic_high_threshold": high_threshold,
            "rule_similarity_threshold": rule_similarity_threshold,
        },
        "summary": {
            "current_rule": {
                **_metrics(rows, "current_rule"),
                "by_category": _category_metrics(rows, "current_rule"),
            },
            "semantic_v1": {
                **_metrics(rows, "semantic_v1"),
                "by_category": _category_metrics(rows, "semantic_v1"),
            },
            "elapsed_s": round(elapsed, 6),
        },
        "rows": rows,
    }


def _decision_digest(result: dict[str, Any]) -> str:
    payload = [
        [
            row["id"],
            row["current_rule"],
            row["semantic_v1"],
            row["goal_similarity"],
            row["anchor_similarity"],
            row["goal_lexical_similarity"],
        ]
        for row in result["rows"]
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DATASET)
    parser.add_argument(
        "--bge-model-dir",
        type=Path,
        default=ROOT / "model_bundle" / "bge_onnx",
    )
    parser.add_argument("--low-threshold", type=float, default=0.50)
    parser.add_argument("--high-threshold", type=float, default=0.78)
    parser.add_argument("--rule-similarity-threshold", type=float, default=0.15)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    results = [
        run_benchmark(
            cases_path=args.cases,
            bge_model_dir=args.bge_model_dir,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            rule_similarity_threshold=args.rule_similarity_threshold,
        )
        for _ in range(args.repeat)
    ]
    digests = [_decision_digest(result) for result in results]
    if len(set(digests)) != 1:
        raise SystemExit(f"non-deterministic benchmark decisions: {digests}")

    result = results[0]
    result["repeat"] = {
        "runs": args.repeat,
        "deterministic": True,
        "decision_sha256": digests[0],
    }
    if args.sweep:
        result["threshold_sweep"] = threshold_sweep(
            load_cases(args.cases),
            result["rows"],
        )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        key: {
            metric: value
            for metric, value in result["summary"][key].items()
            if metric not in {"by_category", "confusion"}
        }
        for key in ("current_rule", "semantic_v1")
    }
    print(json.dumps({**result["repeat"], **summary}, ensure_ascii=False, indent=2))
    if args.sweep:
        print(
            json.dumps(
                {
                    "threshold_sweep_best": result["threshold_sweep"]["top"][0],
                    "threshold_sweep_note": result["threshold_sweep"]["note"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
