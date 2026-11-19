from __future__ import annotations

from squilla_api_router.cost_policy import (
    choose_tier,
    classify_task,
    estimate_input_tokens,
    large_context_floor_tier,
)


def test_cost_policy_can_choose_stronger_tier_when_cheaper() -> None:
    selected, estimates = choose_tier(
        "c1",
        ["c0", "c1", "c2", "c3"],
        100,
        {
            "c1": (5000.0, 0.0, 5),
            "c2": (1000.0, 0.0, 5),
            "c3": (300.0, 0.0, 5),
        },
    )
    assert selected == "c3"
    assert [item.tier for item in estimates] == ["c1", "c2", "c3"]


def test_cost_policy_never_goes_below_quality_floor() -> None:
    selected, estimates = choose_tier(
        "c2",
        ["c0", "c1", "c2", "c3"],
        100,
        {},
    )
    assert selected in {"c2", "c3"}
    assert [item.tier for item in estimates] == ["c2", "c3"]


def test_explicit_output_cap_prevents_false_upshift() -> None:
    selected, _ = choose_tier(
        "c1",
        ["c1", "c2", "c3"],
        100,
        {
            "c1": (5000.0, 0.0, 5),
            "c2": (1000.0, 0.0, 5),
            "c3": (500.0, 0.0, 5),
        },
        output_cap=128,
    )
    assert selected == "c1"


def test_input_estimate_reads_messages_without_mutation() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "name": "read_file"}],
    }
    before = repr(body)
    assert estimate_input_tokens(body) > 0
    assert repr(body) == before


def test_large_context_floor_uses_full_context_thresholds() -> None:
    assert large_context_floor_tier(24_999) is None
    assert large_context_floor_tier(25_000) == "c2"
    assert large_context_floor_tier(80_000) == "c3"



