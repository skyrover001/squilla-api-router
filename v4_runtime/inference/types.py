from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InferenceRequest:
    current_user_text: str
    history_user_texts: list[str]
    prev_assistant_text: str | None
    prev_assistant_usage: dict[str, Any] | None
    prev_route_decisions: list[Any]
    flags_text_override: str | None = None
    context_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureBundle:
    features_390: np.ndarray[Any, Any]
    raw_bge_1536: np.ndarray[Any, Any]
    bge_channels_used: list[str]
    asst_signal_present: bool
    history_user_text_compacted: str | None = None


@dataclass
class HeadOutputs:
    p_main_lgbm: np.ndarray[Any, Any]
    p_aux_lgbm: np.ndarray[Any, Any] | None
    logits_mlp: np.ndarray[Any, Any]
    p_mlp_calibrated: np.ndarray[Any, Any]


@dataclass
class FinalDecision:
    route_class: str
    margin: float
    difficulty_score: float
    flags: dict[str, bool]
    thinking_mode: str
    prompt_policy: str
    aux_downgrade_applied: bool
    sticky_applied: bool


@dataclass
class InferenceResult:
    decision: FinalDecision
    probabilities: dict[str, float]
    aux_decision_probs: dict[str, float] | None
    intermediates: dict[str, object] | None = None
