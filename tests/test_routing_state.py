from __future__ import annotations

from pathlib import Path
import sqlite3
from uuid import uuid4

from squilla_api_router.routing_state import (
    RequestIdentity,
    RoutingStateStore,
    StateConflict,
    decide_intent,
    resolve_request_identity,
)


def _identity(
    *,
    session_id: str = "session-1",
    latest_user: str = "fix the parser",
    last_role: str = "user",
    system_hash: str = "system",
    tools_hash: str = "tools",
) -> RequestIdentity:
    return RequestIdentity(
        session_id=session_id,
        explicit_session=True,
        fingerprint="fingerprint",
        explicit_intent="",
        task_id_hint="",
        system_hash=system_hash,
        tools_hash=tools_hash,
        anchor="fix the parser",
        latest_user_text=latest_user,
        last_role=last_role,
    )


def _new_store() -> tuple[RoutingStateStore, Path]:
    path = Path(__file__).with_name(f".state-{uuid4().hex}.db")
    return RoutingStateStore(path), path


def test_explicit_session_header_wins() -> None:
    identity = resolve_request_identity(
        {"x-squilla-session-id": "agent-session-42"},
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    assert identity.session_id == "agent-session-42"
    assert identity.explicit_session is True


def test_fingerprint_is_stable_for_same_conversation() -> None:
    left = resolve_request_identity(
        {"user-agent": "agent-client"},
        {
            "messages": [
                {"role": "system", "content": "be useful"},
                {"role": "user", "content": "first task"},
                {"role": "assistant", "content": "working"},
                {"role": "user", "content": "continue"},
            ],
            "tools": [{"type": "function", "name": "read_file"}],
        },
    )
    right = resolve_request_identity(
        {"user-agent": "agent-client"},
        {
            "messages": [
                {"role": "system", "content": "be useful"},
                {"role": "user", "content": "first task"},
                {"role": "assistant", "content": "different progress"},
                {"role": "user", "content": "keep going"},
            ],
            "tools": [{"type": "function", "name": "read_file"}],
        },
    )
    assert left.fingerprint == right.fingerprint
    assert left.session_id == right.session_id


def test_new_continue_redo_and_task_changed() -> None:
    store, path = _new_store()
    try:
        identity = _identity()
        first = decide_intent(identity, None)
        assert first.intent == "new_task"

        state = store.put(
            identity,
            task_id="task-1",
            task_anchor=identity.anchor,
            tier="c1",
            route_class="R1",
            model="model-1",
            provider="provider-1",
            api_format="openai_chat",
            base_url="http://backend",
            thinking_mode="T2",
            prompt_policy="P1",
        )
        loaded = store.get(identity.session_id)
        assert loaded == state

        assert (
            decide_intent(
                _identity(latest_user="fix the parser for empty values"), state
            ).intent
            == "continue"
        )
        assert (
            decide_intent(_identity(latest_user="请重新生成"), state).intent == "redo"
        )
        assert (
            decide_intent(
                _identity(latest_user="请继续，把这个翻译改得更正式一些"), state
            ).intent
            == "continue"
        )
        assert (
            decide_intent(
                _identity(latest_user="deploy kubernetes to a new production cluster"),
                state,
            ).intent
            == "task_changed"
        )
    finally:
        path.unlink(missing_ok=True)


def test_tool_result_always_continues_active_task() -> None:
    store, path = _new_store()
    try:
        identity = _identity()
        state = store.put(
            identity,
            task_id="task-1",
            task_anchor=identity.anchor,
            tier="c2",
            route_class="R2",
            model="model-2",
            provider="provider-1",
            api_format="openai_chat",
            base_url="http://backend",
            thinking_mode="T2",
            prompt_policy="P1",
        )
        decision = decide_intent(
            _identity(latest_user="completely unrelated tool payload", last_role="tool"),
            state,
        )
        assert decision.intent == "continue"
        assert decision.reason == "tool_result_continuation"
    finally:
        path.unlink(missing_ok=True)


def test_quality_feedback_forces_redo() -> None:
    store, path = _new_store()
    try:
        identity = _identity()
        state = store.put(
            identity,
            task_id="task-1",
            task_anchor=identity.anchor,
            tier="c1",
            route_class="R1",
            model="model-1",
            provider="provider-1",
            api_format="openai_chat",
            base_url="http://backend",
            thinking_mode="T1",
            prompt_policy="P1",
        )
        quality_identity = resolve_request_identity(
            {
                "x-squilla-session-id": "session-1",
                "x-squilla-quality-feedback": "tool_validation_failed",
            },
            {"messages": [{"role": "user", "content": "fix it"}]},
        )
        decision = decide_intent(quality_identity, state)
        assert decision.intent == "redo"
        assert decision.source == "explicit"
    finally:
        path.unlink(missing_ok=True)


def test_stored_forced_redo_is_returned_by_state_machine() -> None:
    store, path = _new_store()
    try:
        identity = _identity()
        store.put(
            identity,
            task_id="task-1",
            task_anchor=identity.anchor,
            tier="c0",
            route_class="R0",
            model="model-0",
            provider="provider-1",
            api_format="openai_chat",
            base_url="http://backend",
            thinking_mode="T0",
            prompt_policy="P0",
        )
        assert store.mark_forced_intent(
            identity.session_id,
            "redo",
            "automatic_quality_failure:tool_call_invalid_json",
        )
        state = store.get(identity.session_id)
        assert state is not None
        decision = decide_intent(identity, state)
        assert decision.intent == "redo"
        assert "tool_call_invalid_json" in decision.reason
    finally:
        path.unlink(missing_ok=True)


def test_usage_ledger_roundtrip() -> None:
    store, path = _new_store()
    try:
        call_id = store.record_call(
            session_id="session-1",
            task_id="task-1",
            intent="continue",
            tier="c2",
            route_class="R2",
            model="glm",
            provider="primary",
            task_class="code",
            input_tokens=120,
            cached_tokens=80,
            output_tokens=40,
            reasoning_tokens=10,
            latency_s=1.25,
            cost_units=1000,
            usage_source="estimated",
        )
        calls = store.list_calls("session-1")
        assert call_id == calls[0].call_id
        assert calls[0].cached_tokens == 80
        assert calls[0].output_tokens == 40
        assert calls[0].cost_units == 1000
        assert calls[0].usage_source == "estimated"
        assert calls[0].task_class == "code"
        assert store.task_tier_profiles("code")["c2"][2] == 1
    finally:
        path.unlink(missing_ok=True)


def test_state_revision_conflict_and_ttl() -> None:
    store, path = _new_store()
    try:
        identity = _identity()
        first = store.put(
            identity,
            task_id="task-1",
            task_anchor="anchor",
            tier="c0",
            route_class="R0",
            model="m0",
            provider="p",
            api_format="openai_chat",
            base_url="u",
            thinking_mode="T0",
            prompt_policy="P0",
        )
        second = store.put(
            identity,
            task_id="task-1",
            task_anchor="anchor",
            tier="c1",
            route_class="R1",
            model="m1",
            provider="p",
            api_format="openai_chat",
            base_url="u",
            thinking_mode="T1",
            prompt_policy="P1",
            previous=first,
        )
        assert second.revision == 2
        try:
            store.put(
                identity,
                task_id="task-1",
                task_anchor="anchor",
                tier="c0",
                route_class="R0",
                model="m0",
                provider="p",
                api_format="openai_chat",
                base_url="u",
                thinking_mode="T0",
                prompt_policy="P0",
                previous=first,
            )
        except StateConflict:
            pass
        else:
            raise AssertionError("stale revision was not rejected")

        ttl_store = RoutingStateStore(path, ttl_seconds=1)
        connection = sqlite3.connect(path)
        connection.execute(
            "UPDATE routing_state SET updated_at = updated_at - 10 WHERE session_id = ?",
            (identity.session_id,),
        )
        connection.commit()
        connection.close()
        assert ttl_store.get(identity.session_id) is None
    finally:
        path.unlink(missing_ok=True)


def test_repeated_tool_signature_counter() -> None:
    store, path = _new_store()
    try:
        identity = _identity()
        store.put(
            identity,
            task_id="task-1",
            task_anchor="anchor",
            tier="c0",
            route_class="R0",
            model="m0",
            provider="p",
            api_format="openai_chat",
            base_url="u",
            thinking_mode="T0",
            prompt_policy="P0",
        )
        signature = '[["read","{\\"path\\":\\"a.py\\"}"]]'
        assert store.record_tool_signature(identity.session_id, signature) == 1
        assert store.record_tool_signature(identity.session_id, signature) == 2
        assert store.record_tool_signature(identity.session_id, signature) == 3
        assert store.record_tool_signature(identity.session_id, "") == 0
    finally:
        path.unlink(missing_ok=True)

