"""Fail-safe task-boundary policy for session-aware routing.

The policy never edits prompts or conversation content. It only decides whether
the current turn should remain sticky, force fresh model classification, or
start a new logical task.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from squilla_api_router.routing_state import (
    INTENTS,
    IntentDecision,
    RequestIdentity,
    RouteState,
    _CONTINUE_RE,
    _NEW_TASK_RE,
    _REDO_RE,
    text_similarity,
)
from squilla_api_router.v4_runtime.bge_onnx import OnnxBGE


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "model_bundle" / "task_change" / "v1.joblib"
DEFAULT_BGE_DIR = ROOT / "model_bundle" / "bge_onnx"
CONTINUE_REFERENCE_RE = re.compile(
    r"(继续|接着|上面|刚才|前面|这个|该方案|再|补上|加上|加入|统一|保留|"
    r"continue|above|same|go on|include|add|adjust|refine)",
    re.IGNORECASE,
)
TASK_SHIFT_CUE_RE = re.compile(
    r"(另外|顺便|现在|接下来|换(?:个|成|一个)?|改为|重新做(?:一个)?|"
    r"帮我(?:写|做|分析|设计|实现|比较|生成|翻译|整理|安排)|"
    r"new task|another task|instead|switch to|help me (?:write|build|analy[sz]e|"
    r"design|compare|generate|translate|plan))",
    re.IGNORECASE,
)
REFINE_ACTION_RE = re.compile(
    r"(补|增加|加入|添加|检查|评估|比较|说明|统一|保留|删除|修改|调整|优化|"
    r"重点|列出|给出|处理|覆盖|支持|add|include|check|evaluate|compare|"
    r"explain|keep|remove|modify|adjust|optimize|focus|list|handle|cover|support)",
    re.IGNORECASE,
)
NEW_ACTION_RE = re.compile(
    r"(实现|设计|编写|生成|创建|部署|安排|规划|拟定|翻译|整理|总结|写一首|"
    r"写一份|build|create|design|generate|deploy|plan|draft|translate|"
    r"summari[sz]e|prepare|write)",
    re.IGNORECASE,
)
NUMERIC_FEATURE_NAMES = [
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


@dataclass(frozen=True)
class TaskBoundaryDecision:
    intent: str
    source: str
    reason: str
    similarity: float | None = None
    change_probability: float | None = None
    force_reclassify: bool = False
    safe_for_quality: bool = True


def normalize_explicit(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "new": "new_task",
        "continue": "continue",
        "redo": "redo",
        "regenerate": "redo",
        "task_changed": "task_changed",
    }
    return aliases.get(normalized, normalized)


def runtime_case(
    identity: RequestIdentity,
    previous: RouteState,
    *,
    history_user_texts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": identity.session_id,
        "anchor": previous.task_anchor,
        "task_goal": getattr(previous, "task_goal", "") or previous.task_anchor,
        "latest": identity.latest_user_text,
        "last_role": identity.last_role,
        "explicit_intent": identity.explicit_intent,
        "task_id_hint": identity.task_id_hint,
        "forced_intent": previous.forced_intent,
        "system_changed": identity.system_hash != previous.system_hash,
        "tools_changed": identity.tools_hash != previous.tools_hash,
        "previous_task_id": previous.task_id,
        "history_user_texts": list(history_user_texts or [])[-4:],
    }


def hard_rule_decision(case: Mapping[str, Any]) -> TaskBoundaryDecision | None:
    explicit = normalize_explicit(str(case.get("explicit_intent") or ""))
    if explicit in INTENTS:
        return TaskBoundaryDecision(explicit, "explicit", "explicit_control")
    forced = str(case.get("forced_intent") or "")
    if forced in INTENTS:
        return TaskBoundaryDecision(forced, "rule", "forced_intent")
    task_id_hint = str(case.get("task_id_hint") or "")
    if task_id_hint and task_id_hint != str(case.get("previous_task_id") or ""):
        return TaskBoundaryDecision(
            "task_changed",
            "explicit",
            "explicit_task_id_changed",
        )
    if case.get("system_changed"):
        return TaskBoundaryDecision(
            "task_changed",
            "rule",
            "system_prompt_changed",
        )
    if case.get("tools_changed"):
        return TaskBoundaryDecision(
            "task_changed",
            "rule",
            "tool_schema_changed",
        )

    text = str(case.get("latest") or "").strip()
    if _NEW_TASK_RE.search(text):
        return TaskBoundaryDecision(
            "task_changed",
            "rule",
            "explicit_new_task_phrase",
        )
    if _REDO_RE.search(text):
        return TaskBoundaryDecision("redo", "rule", "explicit_redo_phrase")
    if _CONTINUE_RE.search(text):
        return TaskBoundaryDecision("continue", "rule", "continuation_phrase")
    if str(case.get("last_role") or "") in {"tool", "function", "tool_result"}:
        return TaskBoundaryDecision(
            "continue",
            "rule",
            "tool_result_continuation",
        )
    if text and text in {
        str(case.get("anchor") or "").strip(),
        str(case.get("task_goal") or "").strip(),
    }:
        return TaskBoundaryDecision(
            "redo",
            "rule",
            "same_user_message_repeated",
        )
    return None


def _references(case: Mapping[str, Any]) -> list[str]:
    references = [str(case.get("task_goal") or case.get("anchor") or "")]
    history = case.get("history_user_texts") or []
    if isinstance(history, list):
        references.extend(str(item) for item in history[-4:] if str(item).strip())
    return [item for item in references if item]


def semantic_similarities(
    encoder: OnnxBGE,
    case: Mapping[str, Any],
) -> tuple[float, float]:
    latest = str(case.get("latest") or "").strip()
    references = _references(case)
    if not latest or not references:
        return 0.0, 0.0
    vectors = encoder.encode(
        [latest, *references, str(case.get("anchor") or "")],
        batch_size=len(references) + 2,
        convert_to_numpy=True,
    )
    current = vectors[0]
    goal_similarity = float(np.max(vectors[1 : len(references) + 1] @ current))
    anchor_similarity = float(vectors[-1] @ current)
    return goal_similarity, anchor_similarity


def numeric_features(
    case: Mapping[str, Any],
    *,
    goal_similarity: float,
    anchor_similarity: float,
) -> list[float]:
    text = str(case.get("latest") or "")
    goal = str(case.get("task_goal") or case.get("anchor") or "")
    anchor = str(case.get("anchor") or "")
    return [
        goal_similarity,
        anchor_similarity,
        goal_similarity - anchor_similarity,
        text_similarity(goal, text),
        text_similarity(anchor, text),
        min(len(text), 400) / 400.0,
        min(len(goal), 400) / 400.0,
        float(bool(CONTINUE_REFERENCE_RE.search(text))),
        float(bool(TASK_SHIFT_CUE_RE.search(text))),
        float(bool(REFINE_ACTION_RE.search(text))),
        float(bool(NEW_ACTION_RE.search(text))),
        float(text.rstrip().endswith(("?", "？"))),
        text.count("，") / 10.0,
        text.count(",") / 10.0,
        len(text) / max(1, len(goal)),
    ]


def feature_matrix(
    cases: list[Mapping[str, Any]],
    encoder: OnnxBGE,
    vectorizer: Any,
) -> np.ndarray:
    numeric = []
    texts = []
    for case in cases:
        goal_similarity, anchor_similarity = semantic_similarities(encoder, case)
        numeric.append(
            numeric_features(
                case,
                goal_similarity=goal_similarity,
                anchor_similarity=anchor_similarity,
            )
        )
        texts.append(str(case.get("latest") or ""))
    text_matrix = vectorizer.transform(texts).toarray()
    return np.hstack([np.asarray(numeric, dtype=np.float64), text_matrix])


class TaskChangePolicy:
    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_MODEL_PATH,
        bge_dir: str | Path = DEFAULT_BGE_DIR,
        *,
        hard_threshold: float = 0.45,
        soft_threshold: float = 0.20,
    ) -> None:
        model_path = Path(artifact_path)
        checksum_path = model_path.with_suffix(model_path.suffix + ".sha256")
        if checksum_path.exists():
            expected = checksum_path.read_text(encoding="ascii").strip().lower()
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if not expected or actual != expected:
                raise RuntimeError("task-change model checksum mismatch")
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.vectorizer = artifact["vectorizer"]
        self.encoder = OnnxBGE(bge_dir)
        self.hard_threshold = min(max(float(hard_threshold), 0.0), 1.0)
        self.soft_threshold = min(
            max(float(soft_threshold), 0.0),
            self.hard_threshold,
        )
        self.artifact_version = str(artifact.get("version") or "unknown")

    def decide(self, case: Mapping[str, Any]) -> TaskBoundaryDecision:
        hard = hard_rule_decision(case)
        if hard is not None:
            return hard
        goal_similarity, anchor_similarity = semantic_similarities(
            self.encoder,
            case,
        )
        numeric = np.asarray(
            [
                numeric_features(
                    case,
                    goal_similarity=goal_similarity,
                    anchor_similarity=anchor_similarity,
                )
            ],
            dtype=np.float64,
        )
        text = self.vectorizer.transform(
            [str(case.get("latest") or "")]
        ).toarray()
        matrix = np.hstack([numeric, text])
        probability = float(self.model.predict_proba(matrix)[0, 1])
        if probability >= self.hard_threshold:
            return TaskBoundaryDecision(
                "task_changed",
                "hybrid",
                "high_confidence_task_change",
                similarity=goal_similarity,
                change_probability=probability,
            )
        if probability >= self.soft_threshold:
            return TaskBoundaryDecision(
                "continue",
                "hybrid",
                "uncertain_task_boundary_force_reclassify",
                similarity=goal_similarity,
                change_probability=probability,
                force_reclassify=True,
            )
        return TaskBoundaryDecision(
            "continue",
            "hybrid",
            "high_confidence_same_task",
            similarity=goal_similarity,
            change_probability=probability,
        )


def safe_heuristic_decision(decision: IntentDecision) -> TaskBoundaryDecision:
    """Conservative fallback when the local task-boundary model is unavailable."""
    force = decision.reason in {"task_anchor_similarity", "task_anchor_match"}
    return TaskBoundaryDecision(
        decision.intent,
        decision.source,
        decision.reason,
        similarity=decision.similarity,
        force_reclassify=force,
    )
