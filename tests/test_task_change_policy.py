from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from starlette.requests import Request

import app as router_app
from benchmarks.benchmark_task_change import load_cases
from squilla_api_router.routing_state import RouteState
from squilla_api_router.task_change_policy import (
    TaskBoundaryDecision,
    TaskChangePolicy,
    hard_rule_decision,
    safe_heuristic_decision,
)


HOLDOUT = Path(__file__).resolve().parents[1] / "benchmarks" / "task_change_holdout_v2.jsonl"


def _case(case_id: str) -> dict:
    return next(case for case in load_cases(HOLDOUT) if case["id"] == case_id)


def test_contract_change_precedes_continue_phrase() -> None:
    decision = hard_rule_decision(
        {
            "latest": "切换到新的工具集后继续分析。",
            "tools_changed": True,
            "last_role": "user",
        }
    )

    assert decision is not None
    assert decision.intent == "task_changed"
    assert decision.reason == "tool_schema_changed"


def test_safe_policy_hard_changes_soft_routes_and_sticky_turns() -> None:
    policy = TaskChangePolicy()

    confident_change = policy.decide(_case("hold-cross-change-01"))
    uncertain_change = policy.decide(_case("hold-cross-change-03"))

    assert confident_change.intent == "task_changed"
    assert confident_change.change_probability is not None
    assert confident_change.change_probability >= 0.45
    assert uncertain_change.intent == "continue"
    assert uncertain_change.force_reclassify is True
    assert uncertain_change.change_probability is not None
    assert 0.20 <= uncertain_change.change_probability < 0.45


def test_safe_heuristic_fallback_forces_ambiguous_reclassification() -> None:
    from squilla_api_router.routing_state import IntentDecision

    uncertain = safe_heuristic_decision(
        IntentDecision("continue", "hybrid", "task_anchor_match", 0.3)
    )
    explicit = safe_heuristic_decision(
        IntentDecision("redo", "explicit", "explicit_control")
    )

    assert uncertain.force_reclassify is True
    assert explicit.force_reclassify is False


def test_soft_boundary_bypasses_session_stickiness(monkeypatch, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="squilla.router")
    store_path = Path(__file__).with_name(f".soft-boundary-{uuid4().hex}.db")
    store = router_app.RoutingStateStore(store_path)
    previous = store.put(
        router_app.RequestIdentity(
            session_id="soft-boundary",
            explicit_session=True,
            fingerprint="fp",
            explicit_intent="",
            task_id_hint="",
            system_hash="system",
            tools_hash="tools",
            anchor="实现 JSON 读取脚本",
            latest_user_text="实现 JSON 读取脚本",
            last_role="user",
        ),
        task_id="task-1",
        task_anchor="实现 JSON 读取脚本",
        task_goal="实现 JSON 读取脚本",
        tier="c0",
        route_class="R0",
        model="mock-c0",
        provider="mock",
        api_format="openai_chat",
        base_url="http://mock",
        thinking_mode="T0",
        prompt_policy="P0",
    )
    classify_calls = 0

    def classify(*args, **kwargs):
        nonlocal classify_calls
        classify_calls += 1
        return {
            "route_class": "R2",
            "raw_route_class": "R2",
            "confidence": 1.0,
            "thinking_mode": "T2",
            "prompt_policy": "P1",
            "material_tokens": 1,
            "probabilities": {"R2": 1.0},
        }

    class SoftPolicy:
        def decide(self, case):
            return TaskBoundaryDecision(
                "continue",
                "hybrid",
                "uncertain_task_boundary_force_reclassify",
                similarity=0.4,
                change_probability=0.3,
                force_reclassify=True,
            )

    monkeypatch.setattr(router_app, "_ROUTING_STATE_STORE", store)
    monkeypatch.setattr(router_app, "ROUTER_TASK_CHANGE_MODE", "safe_hybrid")
    monkeypatch.setattr(router_app, "_TASK_CHANGE_POLICY", SoftPolicy())
    monkeypatch.setattr(router_app, "_TASK_CHANGE_POLICY_ERROR", "")
    monkeypatch.setattr(router_app, "ROUTER_COST_AWARE_ENABLED", False)
    monkeypatch.setattr(router_app, "ROUTER_SESSION_STICKY", True)
    monkeypatch.setattr(router_app, "task_change_hard_rule", lambda case: None)
    monkeypatch.setattr(router_app, "_classify", classify)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"x-squilla-session-id", b"soft-boundary"),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8021),
            "scheme": "http",
            "query_string": b"",
        }
    )
    request.state.router_principal = router_app.RouterPrincipal(
        tenant_id="",
        is_admin=True,
        auth_kind="open",
    )
    plan = router_app._prepare_session_route(
        request,
        {
            "model": "router",
            "messages": [{"role": "user", "content": "补上测试"}],
        },
        [{"role": "user", "content": "补上测试"}],
        [],
    )

    assert plan.intent == "continue"
    assert plan.force_reclassify is True
    assert plan.sticky is False
    assert plan.task_id == previous.task_id
    assert plan.task_goal == "补上测试"
    assert classify_calls == 1
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[TASK_BOUNDARY]" in messages
    assert "force_reclassify=True" in messages
    assert "[MODEL_SWITCH]" in messages
    store_path.unlink(missing_ok=True)
