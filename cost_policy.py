"""Read-only, price-aware tier selection.

The classifier tier remains a quality floor. This policy may select a more
capable tier when its lower expected output length makes it cheaper overall.
It never changes request content.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_OUTPUT_TOKENS = {
    "c0": 500.0,
    "c1": 3000.0,
    "c2": 1000.0,
    "c3": 800.0,
}


@dataclass(frozen=True)
class TierCostEstimate:
    tier: str
    input_tokens: int
    cached_tokens: int
    expected_output_tokens: float
    expected_cost_units: float
    observations: int


def estimate_input_tokens(body: Mapping[str, Any]) -> int:
    """Conservative read-only estimate of provider-visible input tokens."""
    material = {
        "messages": body.get("messages"),
        "input": body.get("input"),
        "system": body.get("system"),
        "tools": body.get("tools") or body.get("tool"),
    }
    chars = len(
        json.dumps(material, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    # Mixed Chinese/code prompts average fewer than four chars per token. Use a
    # denser estimate to avoid understating input cost.
    return max(1, (chars + 2) // 3)


def classify_task(body: Mapping[str, Any]) -> str:
    """Classify task shape from request text only; no content is modified."""
    text_parts: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        text_parts.append(raw_input)
    elif isinstance(raw_input, list):
        for message in raw_input:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
    text = "\n".join(text_parts).lower()
    if re.search(r"(设计|架构|分布式|缓存|容灾|architecture|design|distributed)", text):
        return "design"
    if re.search(r"(报错|错误|异常|排查|故障|debug|error|exception|traceback|502)", text):
        return "debug"
    if re.search(r"(翻译|translate)", text):
        return "translate"
    if re.search(r"(计算|多少|等于|math|calculate)", text):
        return "math"
    if re.search(r"(代码|函数|脚本|测试|pytest|python|javascript|sql|api|code|function)", text):
        return "code"
    if re.search(r"(解释|区别|什么是|说明|explain|what is|difference)", text):
        return "explain"
    if body.get("tools") or body.get("tool"):
        return "tool"
    return "general"


def price_rates(tier: str) -> tuple[float, float, float]:
    multiplier = {"c0": 1.0, "c1": 4.0, "c2": 10.0, "c3": 20.0}.get(tier, 20.0)
    cache_hit = float(os.environ.get("ROUTER_PRICE_C0_CACHE_HIT", "1"))
    cache_miss = float(os.environ.get("ROUTER_PRICE_C0_CACHE_MISS", "10"))
    output = float(os.environ.get("ROUTER_PRICE_C0_OUTPUT", "40"))
    return cache_hit * multiplier, cache_miss * multiplier, output * multiplier


def estimate_tier_cost(
    tier: str,
    input_tokens: int,
    *,
    average_output_tokens: float | None,
    cache_hit_ratio: float,
    observations: int,
    output_cap: int | None = None,
) -> TierCostEstimate:
    hit, miss, output = price_rates(tier)
    expected_output = max(
        0.0,
        float(
            average_output_tokens
            if average_output_tokens is not None
            else DEFAULT_OUTPUT_TOKENS.get(tier, 1000.0)
        ),
    )
    if isinstance(output_cap, int) and output_cap > 0:
        expected_output = min(expected_output, float(output_cap))
    cached = int(input_tokens * min(max(cache_hit_ratio, 0.0), 1.0))
    cost = (
        cached * hit
        + (input_tokens - cached) * miss
        + expected_output * output
    )
    return TierCostEstimate(
        tier=tier,
        input_tokens=input_tokens,
        cached_tokens=cached,
        expected_output_tokens=round(expected_output, 2),
        expected_cost_units=round(cost, 2),
        observations=observations,
    )


def choose_tier(
    minimum_tier: str,
    candidates: list[str],
    input_tokens: int,
    profiles: Mapping[str, tuple[float | None, float, int]],
    *,
    output_cap: int | None = None,
) -> tuple[str, list[TierCostEstimate]]:
    ordered = sorted(
        [tier for tier in candidates if int(tier[1:]) >= int(minimum_tier[1:])],
        key=lambda tier: int(tier[1:]),
    )
    estimates = []
    for tier in ordered:
        average_output, cache_hit_ratio, observations = profiles.get(
            tier, (None, 0.0, 0)
        )
        estimates.append(
            estimate_tier_cost(
                tier,
                input_tokens,
                average_output_tokens=average_output,
                cache_hit_ratio=cache_hit_ratio,
                observations=observations,
                output_cap=output_cap,
            )
        )
    if not estimates:
        return minimum_tier, []
    # Stable tie-break: prefer lower tier on equal predicted cost.
    selected = min(estimates, key=lambda item: (item.expected_cost_units, item.tier))
    return selected.tier, estimates


def large_context_floor_tier(
    tokens: int,
    *,
    available_tiers: list[str] | tuple[str, ...] = ("c0", "c1", "c2", "c3"),
    r2_tokens: int = 25_000,
    r3_tokens: int = 80_000,
) -> str | None:
    if tokens >= r3_tokens and "c3" in available_tiers:
        return "c3"
    if tokens >= r2_tokens and "c2" in available_tiers:
        return "c2"
    return None

