"""Train and evaluate a small task-boundary classifier on held-out cases.

Training data: ``task_change_cases.jsonl``.
Evaluation data: ``task_change_holdout.jsonl``.

Hard controls (explicit intent, tool result, contract change, repeated prompt)
remain deterministic rules. A local logistic classifier is used only for
ambiguous user turns. No provider or router HTTP call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from benchmarks.benchmark_task_change import (  # noqa: E402
    _category_metrics,
    _identity,
    _metrics,
    _previous,
    _semantic_decision,
    _semantic_similarities,
    _TASK_SHIFT_CUE_RE,
    load_cases,
)
from squilla_api_router.routing_state import (  # noqa: E402
    INTENTS,
    _CONTINUE_RE,
    _NEW_TASK_RE,
    _REDO_RE,
    decide_intent,
    text_similarity,
)
from squilla_api_router.task_change_policy import (  # noqa: E402
    hard_rule_decision as production_hard_rule_decision,
)
from squilla_api_router.v4_runtime.bge_onnx import OnnxBGE  # noqa: E402


HOLDOUT = Path(__file__).with_name("task_change_holdout_v2.jsonl")
TRAIN = Path(__file__).with_name("task_change_train_v2.jsonl")
OUTPUT = Path(__file__).with_name("task_change_hybrid_latest.json")
_CONTINUE_REFERENCE_RE = re.compile(
    r"(继续|接着|上面|刚才|前面|这个|该方案|再|补上|加上|加入|统一|保留|"
    r"continue|above|same|go on|include|add|adjust|refine)",
    re.IGNORECASE,
)
_REFINE_ACTION_RE = re.compile(
    r"(补|增加|加入|添加|检查|评估|比较|说明|统一|保留|删除|修改|调整|优化|"
    r"重点|列出|给出|处理|覆盖|支持|add|include|check|evaluate|compare|"
    r"explain|keep|remove|modify|adjust|optimize|focus|list|handle|cover|support)",
    re.IGNORECASE,
)
_NEW_ACTION_RE = re.compile(
    r"(实现|设计|编写|生成|创建|部署|安排|规划|拟定|翻译|整理|总结|写一首|"
    r"写一份|build|create|design|generate|deploy|plan|draft|translate|"
    r"summari[sz]e|prepare|write)",
    re.IGNORECASE,
)


def _normalize_explicit(case: dict[str, Any]) -> str:
    value = str(case.get("explicit_intent") or "").strip().lower().replace("-", "_")
    aliases = {
        "new": "new_task",
        "continue": "continue",
        "redo": "redo",
        "regenerate": "redo",
        "task_changed": "task_changed",
    }
    return aliases.get(value, value)


def hard_rule_decision(case: dict[str, Any]) -> tuple[str, str] | None:
    """Rules that are sufficiently explicit to bypass the learned classifier."""
    explicit = _normalize_explicit(case)
    if explicit in INTENTS:
        return explicit, "explicit_control"
    if case.get("forced_intent") in INTENTS:
        return str(case["forced_intent"]), "forced_intent"

    previous = _previous(case)
    if case.get("task_id_hint") and case["task_id_hint"] != previous.task_id:
        return "task_changed", "explicit_task_id_changed"

    # Contract changes precede textual continuation phrases. Otherwise a request
    # such as "use the new toolset and continue" is swallowed as Continue.
    if case.get("system_changed"):
        return "task_changed", "system_prompt_changed"
    if case.get("tools_changed"):
        return "task_changed", "tool_schema_changed"

    text = str(case["latest"]).strip()
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
    return None


def _numeric_features(
    case: dict[str, Any],
    *,
    goal_similarity: float,
    anchor_similarity: float,
) -> list[float]:
    text = str(case["latest"])
    goal = str(case.get("task_goal") or case["anchor"])
    return [
        goal_similarity,
        anchor_similarity,
        goal_similarity - anchor_similarity,
        text_similarity(goal, text),
        text_similarity(str(case["anchor"]), text),
        min(len(text), 400) / 400.0,
        min(len(goal), 400) / 400.0,
        float(bool(_CONTINUE_REFERENCE_RE.search(text))),
        float(bool(_TASK_SHIFT_CUE_RE.search(text))),
        float(bool(_REFINE_ACTION_RE.search(text))),
        float(bool(_NEW_ACTION_RE.search(text))),
        float(text.rstrip().endswith(("?", "？"))),
        text.count("，") / 10.0,
        text.count(",") / 10.0,
        len(text) / max(1, len(goal)),
    ]


def _feature_rows(
    cases: list[dict[str, Any]],
    encoder: OnnxBGE,
) -> tuple[np.ndarray, dict[str, Any]]:
    numeric: list[list[float]] = []
    latest_texts: list[str] = []
    row_info: dict[str, float] = {}
    for case in cases:
        goal_similarity, anchor_similarity = _semantic_similarities(encoder, case)
        numeric.append(
            _numeric_features(
                case,
                goal_similarity=goal_similarity,
                anchor_similarity=anchor_similarity,
            )
        )
        latest_texts.append(str(case["latest"]))
        row_info[case["id"]] = goal_similarity
    return np.asarray(numeric, dtype=np.float64), {
        "latest_texts": latest_texts,
        "goal_similarity": row_info,
    }


def _fit_hybrid(
    train_cases: list[dict[str, Any]],
    encoder: OnnxBGE,
    *,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    ambiguous: list[dict[str, Any]] = []
    for case in train_cases:
        if hard_rule_decision(case) is None:
            if case["expected"] not in {"continue", "redo", "task_changed"}:
                continue
            ambiguous.append(case)

    numeric, info = _feature_rows(ambiguous, encoder)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=1,
        max_features=512,
        sublinear_tf=True,
        norm="l2",
    )
    text_matrix = vectorizer.fit_transform(info["latest_texts"])
    matrix = np.hstack([numeric, text_matrix.toarray()])
    labels = np.asarray(
        [
            1 if case["expected"] == "task_changed" else 0
            for case in ambiguous
        ],
        dtype=np.int64,
    )
    model = LogisticRegression(
        C=0.8,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
        max_iter=2000,
    )
    model.fit(matrix, labels)
    folds = min(5, int(np.bincount(labels, minlength=2).min()))
    threshold = 0.5
    if folds >= 2:
        out_of_fold = cross_val_predict(
            LogisticRegression(
                C=0.8,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
                max_iter=2000,
            ),
            matrix,
            labels,
            cv=StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            ),
            method="predict_proba",
        )[:, 1]
        candidates: list[tuple[float, float, int, int]] = []
        for step in range(20, 81):
            candidate = step / 100.0
            predicted = out_of_fold >= candidate
            false_split = int(np.sum((labels == 0) & predicted))
            missed_change = int(np.sum((labels == 1) & ~predicted))
            # Production-safe routing treats a missed task change as a quality
            # failure because it can keep a new task on an underpowered model.
            # A false split is acceptable: it merely re-runs model selection.
            cost = false_split + 20 * missed_change
            candidates.append((float(cost), -candidate, false_split, missed_change))
        _, negative_threshold, _, _ = min(candidates)
        threshold = -negative_threshold
    if threshold_override is not None:
        threshold = min(max(float(threshold_override), 0.0), 1.0)
    numeric_names = [
        "goal_bge",
        "anchor_bge",
        "bge_gap",
        "goal_lexical",
        "anchor_lexical",
        "latest_length",
        "goal_length",
        "continue_reference",
        "task_shift_cue",
        "refine_action",
        "new_action",
        "question",
        "zh_commas",
        "en_commas",
        "length_ratio",
    ]
    vocab = vectorizer.get_feature_names_out()
    feature_names = [*numeric_names, *[f"char:{item}" for item in vocab]]
    coefficients = model.coef_[0]
    top_changed = sorted(
        zip(feature_names, coefficients),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    top_same = sorted(zip(feature_names, coefficients), key=lambda item: item[1])[:15]
    return {
        "ambiguous_train_cases": len(ambiguous),
        "model": model,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "top_changed_features": [
            [name, round(float(weight), 6)] for name, weight in top_changed
        ],
        "top_same_task_features": [
            [name, round(float(weight), 6)] for name, weight in top_same
        ],
        "threshold": threshold,
    }


def _predict_hybrid(
    cases: list[dict[str, Any]],
    fitted: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    predictions: list[str] = []
    change_probabilities: dict[str, float] = {}
    for case in cases:
        rule = hard_rule_decision(case)
        if rule is not None:
            predictions.append(rule[0])
            change_probabilities[case["id"]] = (
                1.0 if rule[0] == "task_changed" else 0.0
            )
            continue
        goal_similarity, anchor_similarity = _semantic_similarities(
            fitted["encoder"],
            case,
        )
        numeric = np.asarray(
            [
                _numeric_features(
                    case,
                    goal_similarity=goal_similarity,
                    anchor_similarity=anchor_similarity,
                )
            ],
            dtype=np.float64,
        )
        text = fitted["vectorizer"].transform([str(case["latest"])]).toarray()
        probability = float(
            fitted["model"].predict_proba(np.hstack([numeric, text]))[0, 1]
        )
        predictions.append(
            "task_changed"
            if probability >= float(fitted["threshold"])
            else "continue"
        )
        change_probabilities[case["id"]] = probability
    return predictions, change_probabilities


def run(
    *,
    train_path: Path = TRAIN,
    holdout_path: Path = HOLDOUT,
    bge_model_dir: Path = ROOT / "model_bundle" / "bge_onnx",
    threshold_override: float | None = None,
) -> dict[str, Any]:
    train_cases = load_cases(train_path)
    holdout_cases = load_cases(holdout_path)
    overlap = {case["id"] for case in train_cases} & {
        case["id"] for case in holdout_cases
    }
    if overlap:
        raise ValueError(f"train/holdout id overlap: {sorted(overlap)}")

    encoder = OnnxBGE(bge_model_dir)
    fitted = _fit_hybrid(
        train_cases,
        encoder,
        threshold_override=threshold_override,
    )
    hybrid_predictions, change_probabilities = _predict_hybrid(
        holdout_cases,
        fitted,
    )

    rows: list[dict[str, Any]] = []
    for case, hybrid in zip(holdout_cases, hybrid_predictions):
        identity = _identity(case)
        previous = _previous(case)
        current = decide_intent(
            identity,
            previous,
            explicit_intent=identity.explicit_intent,
            task_similarity_threshold=0.15,
        )
        goal_similarity, anchor_similarity = _semantic_similarities(encoder, case)
        semantic, semantic_reason = _semantic_decision(
            case,
            goal_similarity,
            low_threshold=0.50,
            high_threshold=0.78,
        )
        probability = change_probabilities[case["id"]]
        boundary_hard = production_hard_rule_decision(case)
        if boundary_hard is not None:
            safe_hybrid = boundary_hard.intent
            safe_force_reclassify = False
        elif probability >= 0.45:
            safe_hybrid = "task_changed"
            safe_force_reclassify = False
        else:
            safe_hybrid = "continue"
            safe_force_reclassify = probability >= 0.20
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected": case["expected"],
                "current_rule": current.intent,
                "semantic_v1": semantic,
                "semantic_reason": semantic_reason,
                "hybrid_v2": hybrid,
                "safe_hybrid": safe_hybrid,
                "safe_force_reclassify": safe_force_reclassify,
                "change_probability": round(
                    probability,
                    6,
                ),
                "goal_similarity": round(goal_similarity, 6),
                "anchor_similarity": round(anchor_similarity, 6),
                "similarity": round(goal_similarity, 6),
            }
        )

    fresh_rows = [row for row in rows if str(row["id"]).startswith("v2-")]
    safe_summary = _metrics(rows, "safe_hybrid")
    safe_summary["unsafe_missed_change_count"] = sum(
        1
        for row in rows
        if row["expected"] == "task_changed"
        and row["safe_hybrid"] != "task_changed"
        and not row["safe_force_reclassify"]
    )
    safe_summary["unsafe_missed_change_rate"] = round(
        safe_summary["unsafe_missed_change_count"]
        / max(
            1,
            sum(row["expected"] == "task_changed" for row in rows),
        ),
        6,
    )
    return {
        "train_dataset": str(train_path),
        "holdout_dataset": str(holdout_path),
        "holdout_cases": len(rows),
        "ambiguous_train_cases": fitted["ambiguous_train_cases"],
        "model": {
            "type": "LogisticRegression",
            "text_features": "char 1-3gram TF-IDF, max 512",
            "numeric_features": 15,
            "threshold": fitted["threshold"],
            "top_changed_features": fitted["top_changed_features"],
            "top_same_task_features": fitted["top_same_task_features"],
        },
        "summary": {
            key: {
                **_metrics(rows, key),
                "by_category": _category_metrics(rows, key),
            }
            for key in ("current_rule", "semantic_v1", "hybrid_v2")
        },
        "safe_hybrid_summary": {
            **safe_summary,
            "by_category": _category_metrics(rows, "safe_hybrid"),
        },
        "fresh_scenario_summary": {
            key: {
                **_metrics(fresh_rows, key),
                "by_category": _category_metrics(fresh_rows, key),
            }
            for key in ("current_rule", "semantic_v1", "hybrid_v2", "safe_hybrid")
        },
        "rows": rows,
    }


def _digest(result: dict[str, Any]) -> str:
    payload = [
        [
            row["id"],
            row["current_rule"],
            row["semantic_v1"],
            row["hybrid_v2"],
            row["change_probability"],
        ]
        for row in result["rows"]
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument(
        "--bge-model-dir",
        type=Path,
        default=ROOT / "model_bundle" / "bge_onnx",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--safe-threshold",
        type=float,
        default=None,
        help="Override the OOF threshold; lower values favor catching task changes.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    results = [
        run(
            train_path=args.train,
            holdout_path=args.holdout,
            bge_model_dir=args.bge_model_dir,
            threshold_override=args.safe_threshold,
        )
        for _ in range(args.repeat)
    ]
    digests = [_digest(result) for result in results]
    if len(set(digests)) != 1:
        raise SystemExit("hybrid benchmark is non-deterministic")
    result = results[0]
    result["repeat"] = {
        "runs": args.repeat,
        "deterministic": True,
        "decision_sha256": digests[0],
    }
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
        for key in ("current_rule", "semantic_v1", "hybrid_v2")
    }
    summary["safe_hybrid"] = {
        metric: value
        for metric, value in result["safe_hybrid_summary"].items()
        if metric
        in {
            "cases",
            "accuracy",
            "macro_f1",
            "false_split_count",
            "false_split_rate",
            "missed_change_count",
            "missed_change_rate",
            "unsafe_missed_change_count",
            "unsafe_missed_change_rate",
        }
    }
    print(
        json.dumps(
            {
                "repeat": result["repeat"],
                "holdout_cases": result["holdout_cases"],
                "ambiguous_train_cases": result["ambiguous_train_cases"],
                "summary": summary,
                "safe_hybrid_quality": {
                    key: value
                    for key, value in result["safe_hybrid_summary"].items()
                    if key
                    in {
                        "unsafe_missed_change_count",
                        "unsafe_missed_change_rate",
                        "false_split_rate",
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
