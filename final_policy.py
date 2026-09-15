from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from squilla_api_router.conversation_context import ConversationContext

_ROUTE_CLASSES = ("R0", "R1", "R2", "R3")
_THINKING_MODE_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

_COMPLAINT_TERMS = (
    "不对",
    "不行",
    "不对劲",
    "还是不对",
    "完全不对",
    "不是这样",
    "你搞错了",
    "你说错了",
    "回答错了",
    "理解错了",
    "搞错重点了",
    "错了",
    "答非所问",
    "没理解",
    "没听懂",
    "太差",
    "太敷衍",
    "敷衍",
    "没用",
    "废话",
    "离谱",
    "乱说",
    "瞎说",
    "胡扯",
    "答得太差",
    "质量太差",
    "不满意",
    "胡说",
    "漏了",
    "遗漏了",
    "没提到",
    "没覆盖",
    "跑题了",
    "偏题了",
    "不是我要的",
    "没按要求",
    "没有按要求",
    "重写",
    "重新来",
    "重新回答",
    "再来一版",
    "换个说法",
    "重新组织",
    "按我说的重来",
    "你没有回答",
    "垃圾",
    "傻逼",
    "sb",
    "蠢",
    "废物",
    "滚",
    "妈的",
    "操",
    "艹",
    "wrong",
    "incorrect",
    "not correct",
    "you are wrong",
    "completely wrong",
    "totally wrong",
    "not what i asked",
    "you misunderstood",
    "that's not right",
    "this is not right",
    "bad answer",
    "terrible answer",
    "awful answer",
    "horrible answer",
    "poor answer",
    "lazy answer",
    "low quality",
    "poor quality",
    "try again",
    "redo",
    "rewrite",
    "start over",
    "answer again",
    "you missed",
    "missed the point",
    "off topic",
    "irrelevant",
    "not helpful",
    "garbage",
    "trash",
    "crap",
    "sucks",
    "stupid",
    "idiot",
    "moron",
    "dumb",
    "pathetic",
    "ridiculous",
    "fuck",
    "fucking",
    "shit",
    "damn",
    "wtf",
    "asshole",
    "bullshit",
    "nonsense",
    "useless",
)


@dataclass(frozen=True)
class _PolicyProfile:
    version: str = "v4-phase3-standalone-policy-1"
    default_route_class: str = "R1"
    confidence_threshold: float = 0.5
    confidence_high_tier_margin: float = 0.05
    complaint_upgrade_steps: int = 1
    complaint_upgrade_max_chars: int = 160
    complaint_terms: tuple[str, ...] = _COMPLAINT_TERMS
    large_context_r2_floor_tokens: int = 25_000
    large_context_r3_floor_tokens: int = 80_000
    large_context_r3_context_ratio: float = 0.4
    context_window_tokens: int = 200_000


_PROFILE = _PolicyProfile()
PROFILE_VERSION = _PROFILE.version


@dataclass(frozen=True)
class FinalPolicyResult:
    final_decision: dict[str, str]
    decision_trace: list[dict[str, object]]


def profile_config_digest() -> str:
    serialized = json.dumps(
        asdict(_PROFILE),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _route_index(route_class: str) -> int:
    return _ROUTE_CLASSES.index(route_class)


def _stage_entry(
    stage: str,
    *,
    applied: bool,
    from_route: str,
    to_route: str,
    facts: dict[str, object],
    reason_code: str,
    skipped: bool = False,
) -> dict[str, object]:
    return {
        "stage": stage,
        "status": "skipped" if skipped else "applied" if applied else "not_applied",
        "applied": applied,
        "from": from_route,
        "to": to_route,
        "facts": facts,
        "reason_code": reason_code,
    }


def _confidence_gate(
    route_class: str,
    confidence: float,
    profile: _PolicyProfile,
) -> tuple[str, dict[str, object]]:
    default_index = _route_index(profile.default_route_class)
    route_index = _route_index(route_class)
    cutoff = profile.confidence_threshold
    if route_index > default_index:
        cutoff -= profile.confidence_high_tier_margin

    to_route = route_class
    if confidence < cutoff and route_class != profile.default_route_class:
        to_route = profile.default_route_class
    applied = to_route != route_class
    if applied:
        reason_code = "confidence_below_cutoff"
    elif route_class == profile.default_route_class:
        reason_code = "already_default_route"
    else:
        reason_code = "confidence_at_or_above_cutoff"
    return to_route, _stage_entry(
        "confidence_gate",
        applied=applied,
        from_route=route_class,
        to_route=to_route,
        facts={
            "confidence": confidence,
            "threshold": profile.confidence_threshold,
            "high_tier_margin": profile.confidence_high_tier_margin,
            "cutoff": cutoff,
            "default_route_class": profile.default_route_class,
        },
        reason_code=reason_code,
    )


def _complaint_upgrade(
    route_class: str,
    *,
    message: str,
    pre_confidence_route_class: str,
    previous_route_class: str | None,
    profile: _PolicyProfile,
) -> tuple[str, dict[str, object], bool]:
    stripped_message = message.strip()
    message_chars = len(stripped_message)
    common_facts: dict[str, object] = {
        "message_chars": message_chars,
        "max_chars": profile.complaint_upgrade_max_chars,
        "matched_terms_count": 0,
        "upgrade_steps": profile.complaint_upgrade_steps,
        "pre_confidence_route_class": pre_confidence_route_class,
    }
    if previous_route_class is not None:
        common_facts["previous_route_class"] = previous_route_class
    if message_chars > profile.complaint_upgrade_max_chars:
        return route_class, _stage_entry(
            "complaint_upgrade",
            applied=False,
            from_route=route_class,
            to_route=route_class,
            facts=common_facts,
            reason_code="complaint_message_too_long",
            skipped=True,
        ), False

    lowered = stripped_message.lower()
    matched_count = sum(term in lowered for term in profile.complaint_terms)
    common_facts["matched_terms_count"] = matched_count
    if matched_count == 0:
        return route_class, _stage_entry(
            "complaint_upgrade",
            applied=False,
            from_route=route_class,
            to_route=route_class,
            facts=common_facts,
            reason_code="complaint_not_detected",
        ), False

    starting_index = max(_route_index(route_class), _route_index(pre_confidence_route_class))
    if previous_route_class is not None:
        starting_index = max(starting_index, _route_index(previous_route_class))
    target_index = min(
        starting_index + max(profile.complaint_upgrade_steps, 0),
        len(_ROUTE_CLASSES) - 1,
    )
    to_route = _ROUTE_CLASSES[target_index]
    applied = to_route != route_class
    return to_route, _stage_entry(
        "complaint_upgrade",
        applied=applied,
        from_route=route_class,
        to_route=to_route,
        facts=common_facts,
        reason_code="complaint_upgrade" if applied else "already_highest_route",
    ), True


def _anti_downgrade(
    route_class: str,
    previous_route_class: str | None,
) -> tuple[str, dict[str, object]]:
    if previous_route_class is None:
        return route_class, _stage_entry(
            "anti_downgrade",
            applied=False,
            from_route=route_class,
            to_route=route_class,
            facts={"history_available": False},
            reason_code="route_history_missing",
            skipped=True,
        )

    applied = _route_index(previous_route_class) > _route_index(route_class)
    to_route = previous_route_class if applied else route_class
    return to_route, _stage_entry(
        "anti_downgrade",
        applied=applied,
        from_route=route_class,
        to_route=to_route,
        facts={"previous_route_class": previous_route_class},
        reason_code="previous_route_floor" if applied else "previous_route_not_higher",
    )


def _large_context_floor(
    route_class: str,
    context: ConversationContext,
    profile: _PolicyProfile,
) -> tuple[str, dict[str, object]]:
    material_tokens = context.material_estimated_tokens
    ratio_floor_tokens = int(
        profile.context_window_tokens * profile.large_context_r3_context_ratio
    )
    minimum_route_class: str | None = None
    if (
        material_tokens >= profile.large_context_r3_floor_tokens
        or material_tokens >= ratio_floor_tokens
    ):
        minimum_route_class = "R3"
    elif material_tokens >= profile.large_context_r2_floor_tokens:
        minimum_route_class = "R2"

    facts: dict[str, object] = {
        "material_tokens": material_tokens,
        "context_window_tokens": profile.context_window_tokens,
        "r2_floor_tokens": profile.large_context_r2_floor_tokens,
        "r3_floor_tokens": profile.large_context_r3_floor_tokens,
        "r3_context_ratio": profile.large_context_r3_context_ratio,
        "r3_ratio_floor_tokens": ratio_floor_tokens,
    }
    if minimum_route_class is None:
        return route_class, _stage_entry(
            "large_context_floor",
            applied=False,
            from_route=route_class,
            to_route=route_class,
            facts=facts,
            reason_code="below_large_context_floor",
        )

    facts["minimum_route_class"] = minimum_route_class
    applied = _route_index(minimum_route_class) > _route_index(route_class)
    to_route = minimum_route_class if applied else route_class
    return to_route, _stage_entry(
        "large_context_floor",
        applied=applied,
        from_route=route_class,
        to_route=to_route,
        facts=facts,
        reason_code=(
            "large_context_floor" if applied else "route_at_or_above_large_context_floor"
        ),
    )


def _reconcile_controller(
    route_class: str,
    thinking_mode: str,
    prompt_policy: str,
    *,
    complaint_detected: bool,
) -> tuple[str, str]:
    minimum_thinking = {"R1": "T1", "R2": "T2", "R3": "T3"}.get(route_class)
    if (
        minimum_thinking is not None
        and _THINKING_MODE_ORDER[thinking_mode] < _THINKING_MODE_ORDER[minimum_thinking]
    ):
        thinking_mode = minimum_thinking
    if prompt_policy == "P0" and (route_class in {"R2", "R3"} or complaint_detected):
        prompt_policy = "P1"
    if thinking_mode in {"T2", "T3"} and prompt_policy == "P0":
        prompt_policy = "P1"
    return thinking_mode, prompt_policy


def apply_final_policy(
    *,
    route_class: str,
    confidence: float,
    thinking_mode: str,
    prompt_policy: str,
    context: ConversationContext,
) -> FinalPolicyResult:
    trace: list[dict[str, object]] = []
    current_route, entry = _confidence_gate(route_class, confidence, _PROFILE)
    trace.append(entry)

    previous_route_class = (
        context.route_history[-1].route_class if context.route_history else None
    )
    current_route, entry, complaint_detected = _complaint_upgrade(
        current_route,
        message=context.current_user_text or "",
        pre_confidence_route_class=route_class,
        previous_route_class=previous_route_class,
        profile=_PROFILE,
    )
    trace.append(entry)

    current_route, entry = _anti_downgrade(current_route, previous_route_class)
    trace.append(entry)
    current_route, entry = _large_context_floor(current_route, context, _PROFILE)
    trace.append(entry)

    thinking_mode, prompt_policy = _reconcile_controller(
        current_route,
        thinking_mode,
        prompt_policy,
        complaint_detected=complaint_detected,
    )
    return FinalPolicyResult(
        final_decision={
            "route_class": current_route,
            "thinking_mode": thinking_mode,
            "prompt_policy": prompt_policy,
        },
        decision_trace=trace,
    )

