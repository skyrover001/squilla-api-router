from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from squilla_api_router.v4_runtime.flags import RoutingFlags

ROUTE_CLASSES = ["R0", "R1", "R2", "R3"]
_CLASS_TO_IDX = {route_class: index for index, route_class in enumerate(ROUTE_CLASSES)}


class RouteDecision(Protocol):
    route_class: str


def _apply_margin_upgrade(
    route_class: str,
    margin: float,
    config: dict[str, Any],
) -> str:
    threshold = float(config.get("thresholds", {}).get("margin_upgrade", 0.15))
    if margin < threshold:
        index = _CLASS_TO_IDX[route_class]
        if index < len(ROUTE_CLASSES) - 1:
            return ROUTE_CLASSES[index + 1]
    return route_class


def _apply_r1_rescue(
    route_class: str,
    probabilities: np.ndarray[Any, Any],
    config: dict[str, Any],
) -> str:
    rescue = config.get("thresholds", {}).get("r1_rescue", {})
    maximum_gap = float(rescue.get("from_r0_max_gap", 0.20))
    if route_class == "R0":
        r1_probability = float(probabilities[1])
        r0_probability = float(probabilities[0])
        if r0_probability - r1_probability < maximum_gap:
            return "R1"
    return route_class


def _apply_flag_overrides(
    route_class: str,
    flags: RoutingFlags,
    _config: dict[str, Any],
) -> str:
    index = _CLASS_TO_IDX[route_class]
    if flags.high_risk:
        index = max(index, _CLASS_TO_IDX["R2"])
    if flags.debug and flags.long_context:
        index = max(index, _CLASS_TO_IDX["R2"])
    if flags.repo_arch:
        index = max(index, _CLASS_TO_IDX["R1"])
    return ROUTE_CLASSES[index]


def _derive_thinking_mode(
    route_class: str,
    margin: float,
    flags: RoutingFlags,
    config: dict[str, Any],
) -> str:
    rules = config.get("thinking_mode_rules", {})
    if route_class == "R3":
        return "T3"
    t3_rule = rules.get("T3", {})
    t3_flags = t3_rule.get("flags", ["debug", "long_context", "high_risk"])
    minimum_class = str(t3_rule.get("min_class", "R2"))
    if _CLASS_TO_IDX[route_class] >= _CLASS_TO_IDX.get(minimum_class, 2):
        for flag_name in t3_flags:
            if bool(getattr(flags, str(flag_name), False)):
                return "T3"
    t0_rule = rules.get("T0", {})
    t0_maximum = str(t0_rule.get("max_class", "R0"))
    if (
        _CLASS_TO_IDX[route_class] <= _CLASS_TO_IDX.get(t0_maximum, 0)
        and margin >= float(t0_rule.get("min_margin", 0.5))
    ):
        return "T0"
    t1_rule = rules.get("T1", {})
    t1_maximum = str(t1_rule.get("max_class", "R1"))
    if (
        _CLASS_TO_IDX[route_class] <= _CLASS_TO_IDX.get(t1_maximum, 1)
        and margin >= float(t1_rule.get("min_margin", 0.4))
    ):
        return "T1"
    return "T2"


def _derive_prompt_policy(
    difficulty_score: float,
    margin: float,
    flags: RoutingFlags,
    config: dict[str, Any],
) -> str:
    policies = config.get("prompt_policies", {})
    p2_conditions = policies.get("P2", {}).get("conditions", {})
    policy_flags = p2_conditions.get(
        "any_flag",
        ["high_risk", "long_context", "debug", "strict_format"],
    )
    for flag_name in policy_flags:
        if bool(getattr(flags, str(flag_name), False)):
            return "P2"
    p0_conditions = policies.get("P0", {}).get("conditions", {})
    blocking_flags = p0_conditions.get(
        "no_flags",
        ["high_risk", "strict_format", "debug"],
    )
    has_blocking_flag = any(
        bool(getattr(flags, str(flag_name), False)) for flag_name in blocking_flags
    )
    if (
        difficulty_score <= float(p0_conditions.get("max_difficulty", 0.8))
        and margin >= float(p0_conditions.get("min_margin", 0.4))
        and not has_blocking_flag
    ):
        return "P0"
    return "P1"


def _apply_sticky_tier(
    predicted_class: str,
    _probabilities: np.ndarray[Any, Any],
    history: list[RouteDecision] | None,
    config: dict[str, Any],
) -> str:
    if not history or not config.get("thresholds", {}).get("kv_cache_aware", False):
        return predicted_class
    previous_class = history[-1].route_class
    if _CLASS_TO_IDX[previous_class] > _CLASS_TO_IDX[predicted_class]:
        return previous_class
    return predicted_class

