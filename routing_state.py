"""Transparent session/task routing state.

This module never edits prompts or tool definitions. It only derives a stable
session identity, classifies the turn intent, and persists the model selected
for the active task so ordinary tool loops can stay on the same deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


INTENTS = {"new_task", "continue", "redo", "task_changed"}
_REDO_RE = re.compile(
    r"(重新生成|重做|再来一次|再试一次|重新回答|重新执行|retry|regenerate|try again|do it again)",
    re.IGNORECASE,
)
_NEW_TASK_RE = re.compile(
    r"(新任务|另一个任务|换个任务|换一个任务|现在做(?:另|别的)|另外做一个|"
    r"new task|another task|switch task|different task)",
    re.IGNORECASE,
)
_CONTINUE_RE = re.compile(
    r"(继续|接着|基于上面|基于刚才|同一个任务|这个任务|这个翻译|改成|修改一下|调整一下|"
    r"continue|keep going|same task|based on the above|adjust|revise)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]", re.IGNORECASE)


def _first_header(headers: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = headers.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _messages(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = body.get("messages")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    raw = body.get("input")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    return []


def _system_material(body: Mapping[str, Any]) -> Any:
    if "system" in body:
        return body.get("system")
    return [
        {"role": item.get("role"), "content": item.get("content")}
        for item in _messages(body)
        if item.get("role") in {"system", "developer"}
    ]


def _first_user_text(body: Mapping[str, Any]) -> str:
    for item in _messages(body):
        if item.get("role") == "user":
            text = _content_text(item.get("content"))
            if text:
                return text
    return ""


def _last_user_text(body: Mapping[str, Any]) -> str:
    for item in reversed(_messages(body)):
        if item.get("role") == "user":
            text = _content_text(item.get("content"))
            if text:
                return text
    return ""


def _last_role(body: Mapping[str, Any]) -> str:
    messages = _messages(body)
    return str(messages[-1].get("role") or "") if messages else ""


def _tokens(text: str) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    words = set(_TOKEN_RE.findall(compact))
    cjk = "".join(ch for ch in compact if "\u3400" <= ch <= "\u9fff")
    words.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    return words


def text_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


@dataclass(frozen=True)
class RequestIdentity:
    session_id: str
    explicit_session: bool
    fingerprint: str
    explicit_intent: str
    task_id_hint: str
    system_hash: str
    tools_hash: str
    anchor: str
    latest_user_text: str
    last_role: str


@dataclass(frozen=True)
class RouteState:
    session_id: str
    task_id: str
    task_anchor: str
    system_hash: str
    tools_hash: str
    tier: str
    route_class: str
    model: str
    provider: str
    api_format: str
    base_url: str
    thinking_mode: str
    prompt_policy: str
    forced_intent: str
    forced_reason: str
    last_tool_signature: str
    repeated_tool_count: int
    revision: int
    turn_count: int
    updated_at: float


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    source: str
    reason: str
    similarity: float | None = None


@dataclass(frozen=True)
class RouteCall:
    call_id: int
    session_id: str
    task_id: str
    timestamp: float
    intent: str
    tier: str
    route_class: str
    model: str
    provider: str
    task_class: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_s: float
    cost_units: int


@dataclass(frozen=True)
class RouteAttempt:
    attempt_id: int
    session_id: str
    task_id: str
    timestamp: float
    intent: str
    tier: str
    route_class: str
    model: str
    provider: str
    success: int
    status_code: int | None
    error_type: str
    latency_s: float
    cooldown_s: float


class StateConflict(RuntimeError):
    pass


def resolve_request_identity(
    headers: Mapping[str, str], body: Mapping[str, Any]
) -> RequestIdentity:
    explicit_session = _first_header(
        headers,
        (
            "x-squilla-session-id",
            "x-session-id",
            "conversation-id",
            "conversation_id",
        ),
    )
    if not explicit_session:
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            explicit_session = str(
                metadata.get("session_id") or metadata.get("conversation_id") or ""
            ).strip()
        explicit_session = explicit_session or str(
            body.get("conversation_id") or body.get("session_id") or ""
        ).strip()

    explicit_intent = _first_header(
        headers, ("x-squilla-intent", "x-router-intent")
    )
    quality_feedback = _first_header(
        headers,
        (
            "x-squilla-quality-feedback",
            "x-squilla-retry-reason",
            "x-router-quality-retry",
        ),
    )
    task_id_hint = _first_header(
        headers, ("x-squilla-task-id", "x-task-id")
    )
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        explicit_intent = explicit_intent or str(
            metadata.get("intent") or metadata.get("routing_intent") or ""
        ).strip()
        task_id_hint = task_id_hint or str(
            metadata.get("task_id") or ""
        ).strip()
        quality_feedback = quality_feedback or str(
            metadata.get("quality_feedback")
            or metadata.get("retry_reason")
            or ""
        ).strip()

    if quality_feedback and quality_feedback.lower() not in {
        "0",
        "false",
        "none",
        "ok",
        "success",
    }:
        explicit_intent = "redo"

    system_hash = _json_hash(_system_material(body))
    tools_hash = _json_hash(body.get("tools") or body.get("tool") or [])
    first_user = _first_user_text(body)
    latest_user = _last_user_text(body)
    client = _first_header(headers, ("x-squilla-client", "user-agent"))
    fingerprint_material = {
        "client": client,
        "system": system_hash,
        "tools": tools_hash,
        "first_user": first_user[:1000],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    session_id = explicit_session or f"fp:{fingerprint}"
    return RequestIdentity(
        session_id=session_id,
        explicit_session=bool(explicit_session),
        fingerprint=fingerprint,
        explicit_intent=explicit_intent,
        task_id_hint=task_id_hint,
        system_hash=system_hash,
        tools_hash=tools_hash,
        anchor=first_user or latest_user,
        latest_user_text=latest_user,
        last_role=_last_role(body),
    )


def decide_intent(
    identity: RequestIdentity,
    previous: RouteState | None,
    *,
    explicit_intent: str = "",
    task_similarity_threshold: float = 0.15,
) -> IntentDecision:
    explicit_intent = explicit_intent or identity.explicit_intent
    normalized_intent = explicit_intent.strip().lower().replace("-", "_")
    aliases = {
        "new": "new_task",
        "continue": "continue",
        "redo": "redo",
        "regenerate": "redo",
        "task_changed": "task_changed",
    }
    normalized_intent = aliases.get(normalized_intent, normalized_intent)
    if normalized_intent in INTENTS:
        return IntentDecision(normalized_intent, "explicit", "explicit_control")

    if previous is None:
        return IntentDecision("new_task", "rule", "no_active_task")
    if previous.forced_intent in INTENTS:
        return IntentDecision(previous.forced_intent, "rule", previous.forced_reason or "stored_quality_feedback")
    if identity.task_id_hint and identity.task_id_hint != previous.task_id:
        return IntentDecision("task_changed", "explicit", "explicit_task_id_changed")

    text = identity.latest_user_text.strip()
    if _NEW_TASK_RE.search(text):
        return IntentDecision("task_changed", "rule", "explicit_new_task_phrase")
    if _REDO_RE.search(text):
        return IntentDecision("redo", "rule", "explicit_redo_phrase")
    if _CONTINUE_RE.search(text):
        return IntentDecision("continue", "rule", "continuation_phrase")

    if identity.system_hash != previous.system_hash:
        return IntentDecision("task_changed", "rule", "system_prompt_changed")
    if identity.tools_hash != previous.tools_hash:
        return IntentDecision("task_changed", "rule", "tool_schema_changed")

    if identity.last_role in {"tool", "function", "tool_result"}:
        return IntentDecision("continue", "rule", "tool_result_continuation")
    if text and text == previous.task_anchor.strip():
        return IntentDecision("redo", "rule", "same_user_message_repeated")

    similarity = text_similarity(previous.task_anchor, text)
    if text and len(text) >= 12 and similarity < task_similarity_threshold:
        return IntentDecision(
            "task_changed",
            "hybrid",
            "task_anchor_similarity",
            similarity=similarity,
        )
    return IntentDecision(
        "continue",
        "hybrid",
        "task_anchor_match",
        similarity=similarity,
    )


class RoutingStateStore:
    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        ttl_seconds: float = 86400.0,
    ) -> None:
        default_path = Path.cwd() / ".squilla_router_state.db"
        self.path = Path(path or os.environ.get("ROUTER_STATE_DB") or default_path)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_state (
                    session_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_anchor TEXT NOT NULL,
                    system_hash TEXT NOT NULL,
                    tools_hash TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    route_class TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    api_format TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    thinking_mode TEXT NOT NULL,
                    prompt_policy TEXT NOT NULL,
                    forced_intent TEXT NOT NULL DEFAULT '',
                    forced_reason TEXT NOT NULL DEFAULT '',
                    last_tool_signature TEXT NOT NULL DEFAULT '',
                    repeated_tool_count INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    turn_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            state_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(routing_state)"
                ).fetchall()
            }
            if "forced_intent" not in state_columns:
                connection.execute(
                    "ALTER TABLE routing_state ADD COLUMN forced_intent TEXT NOT NULL DEFAULT ''"
                )
            if "forced_reason" not in state_columns:
                connection.execute(
                    "ALTER TABLE routing_state ADD COLUMN forced_reason TEXT NOT NULL DEFAULT ''"
                )
            if "revision" not in state_columns:
                connection.execute(
                    "ALTER TABLE routing_state ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            if "last_tool_signature" not in state_columns:
                connection.execute(
                    "ALTER TABLE routing_state ADD COLUMN last_tool_signature TEXT NOT NULL DEFAULT ''"
                )
            if "repeated_tool_count" not in state_columns:
                connection.execute(
                    "ALTER TABLE routing_state ADD COLUMN repeated_tool_count INTEGER NOT NULL DEFAULT 0"
                )
            if self.ttl_seconds:
                connection.execute(
                    "DELETE FROM routing_state WHERE updated_at < ?",
                    (time.time() - self.ttl_seconds,),
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_calls (
                    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    intent TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    route_class TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    task_class TEXT NOT NULL DEFAULT 'unknown',
                    input_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    latency_s REAL NOT NULL,
                    cost_units INTEGER NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(routing_calls)"
                ).fetchall()
            }
            if "task_class" not in columns:
                connection.execute(
                    "ALTER TABLE routing_calls ADD COLUMN task_class TEXT NOT NULL DEFAULT 'unknown'"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_calls_session
                ON routing_calls(session_id, timestamp)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    intent TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    route_class TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    status_code INTEGER,
                    error_type TEXT NOT NULL,
                    latency_s REAL NOT NULL,
                    cooldown_s REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_attempts_session
                ON routing_attempts(session_id, timestamp)
                """
            )

    def get(self, session_id: str) -> RouteState | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM routing_state WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row and self.ttl_seconds and row["updated_at"] < time.time() - self.ttl_seconds:
                connection.execute(
                    "DELETE FROM routing_state WHERE session_id = ?", (session_id,)
                )
                connection.commit()
                return None
        return RouteState(**dict(row)) if row else None

    def delete(self, session_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM routing_state WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    def purge_expired(self) -> int:
        if not self.ttl_seconds:
            return 0
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM routing_state WHERE updated_at < ?",
                (time.time() - self.ttl_seconds,),
            )
        return int(cursor.rowcount)

    def put(
        self,
        identity: RequestIdentity,
        *,
        task_id: str,
        task_anchor: str,
        tier: str,
        route_class: str,
        model: str,
        provider: str,
        api_format: str,
        base_url: str,
        thinking_mode: str,
        prompt_policy: str,
        previous: RouteState | None = None,
    ) -> RouteState:
        turn_count = (previous.turn_count + 1) if previous else 1
        updated_at = time.time()
        revision = (previous.revision + 1) if previous else 1
        with closing(self._connect()) as connection, connection:
            if previous is None:
                try:
                    connection.execute(
                        """
                        INSERT INTO routing_state (
                            session_id, task_id, task_anchor, system_hash, tools_hash,
                            tier, route_class, model, provider, api_format, base_url,
                            thinking_mode, prompt_policy, forced_intent,
                            forced_reason, revision, turn_count, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', 1, ?, ?)
                        """,
                        (
                            identity.session_id,
                            task_id,
                            task_anchor,
                            identity.system_hash,
                            identity.tools_hash,
                            tier,
                            route_class,
                            model,
                            provider,
                            api_format,
                            base_url,
                            thinking_mode,
                            prompt_policy,
                            turn_count,
                            updated_at,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise StateConflict(identity.session_id) from exc
            else:
                cursor = connection.execute(
                    """
                    UPDATE routing_state SET
                        task_id = ?, task_anchor = ?, system_hash = ?,
                        tools_hash = ?, tier = ?, route_class = ?, model = ?,
                        provider = ?, api_format = ?, base_url = ?,
                        thinking_mode = ?, prompt_policy = ?, revision = ?,
                        turn_count = ?, updated_at = ?
                    WHERE session_id = ? AND revision = ?
                    """,
                    (
                        task_id,
                        task_anchor,
                        identity.system_hash,
                        identity.tools_hash,
                        tier,
                        route_class,
                        model,
                        provider,
                        api_format,
                        base_url,
                        thinking_mode,
                        prompt_policy,
                        revision,
                        turn_count,
                        updated_at,
                        identity.session_id,
                        previous.revision,
                    ),
                )
                if cursor.rowcount == 0:
                    raise StateConflict(identity.session_id)
        return RouteState(
            session_id=identity.session_id,
            task_id=task_id,
            task_anchor=task_anchor,
            system_hash=identity.system_hash,
            tools_hash=identity.tools_hash,
            tier=tier,
            route_class=route_class,
            model=model,
            provider=provider,
            api_format=api_format,
            base_url=base_url,
            thinking_mode=thinking_mode,
            prompt_policy=prompt_policy,
            forced_intent=previous.forced_intent if previous else "",
            forced_reason=previous.forced_reason if previous else "",
            last_tool_signature=(
                previous.last_tool_signature
                if previous and previous.task_id == task_id
                else ""
            ),
            repeated_tool_count=(
                previous.repeated_tool_count
                if previous and previous.task_id == task_id
                else 0
            ),
            revision=revision,
            turn_count=turn_count,
            updated_at=updated_at,
        )

    def mark_forced_intent(
        self, session_id: str, intent: str, reason: str
    ) -> bool:
        if intent not in INTENTS:
            raise ValueError(f"invalid intent: {intent}")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE routing_state
                SET forced_intent = ?, forced_reason = ?, updated_at = ?,
                    revision = revision + 1
                WHERE session_id = ?
                """,
                (intent, reason[:500], time.time(), session_id),
            )
        return cursor.rowcount > 0

    def clear_forced_intent(
        self, session_id: str, expected_reason: str | None = None
    ) -> bool:
        where = "session_id = ?"
        args: list[Any] = [time.time(), session_id]
        if expected_reason is not None:
            where += " AND forced_reason = ?"
            args.append(expected_reason)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                f"""
                UPDATE routing_state
                SET forced_intent = '', forced_reason = '', updated_at = ?,
                    revision = revision + 1
                WHERE {where}
                """,
                tuple(args),
            )
        return cursor.rowcount > 0

    def record_tool_signature(self, session_id: str, signature: str) -> int:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT last_tool_signature, repeated_tool_count FROM routing_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return 0
            if not signature:
                connection.execute(
                    """
                    UPDATE routing_state
                    SET last_tool_signature = '', repeated_tool_count = 0,
                        revision = revision + 1, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (time.time(), session_id),
                )
                return 0
            count = (
                int(row["repeated_tool_count"]) + 1
                if row["last_tool_signature"] == signature
                else 1
            )
            connection.execute(
                """
                UPDATE routing_state
                SET last_tool_signature = ?, repeated_tool_count = ?,
                    revision = revision + 1, updated_at = ?
                WHERE session_id = ?
                """,
                (signature, count, time.time(), session_id),
            )
        return count
    def record_call(
        self,
        *,
        session_id: str,
        task_id: str,
        intent: str,
        tier: str,
        route_class: str,
        model: str,
        provider: str,
        task_class: str = "unknown",
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        latency_s: float,
        cost_units: int,
    ) -> int:
        timestamp = time.time()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO routing_calls (
                    session_id, task_id, timestamp, intent, tier, route_class,
                    model, provider, task_class, input_tokens, cached_tokens, output_tokens,
                    reasoning_tokens, latency_s, cost_units
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    task_id,
                    timestamp,
                    intent,
                    tier,
                    route_class,
                    model,
                    provider,
                    task_class,
                    input_tokens,
                    cached_tokens,
                    output_tokens,
                    reasoning_tokens,
                    latency_s,
                    cost_units,
                ),
            )
        return int(cursor.lastrowid)

    def list_calls(self, session_id: str) -> list[RouteCall]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM routing_calls
                WHERE session_id = ?
                ORDER BY call_id
                """,
                (session_id,),
            ).fetchall()
        return [RouteCall(**dict(row)) for row in rows]

    def tier_profiles(self) -> dict[str, tuple[float | None, float, int]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    tier,
                    AVG(output_tokens) AS avg_output,
                    AVG(
                        CASE
                            WHEN input_tokens > 0
                            THEN CAST(cached_tokens AS REAL) / input_tokens
                            ELSE 0.0
                        END
                    ) AS cache_hit_ratio,
                    COUNT(*) AS observations
                FROM routing_calls
                GROUP BY tier
                """
            ).fetchall()
        return {
            str(row["tier"]): (
                float(row["avg_output"]) if row["avg_output"] is not None else None,
                float(row["cache_hit_ratio"] or 0.0),
                int(row["observations"] or 0),
            )
            for row in rows
        }

    def task_tier_profiles(
        self, task_class: str
    ) -> dict[str, tuple[float | None, float, int]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    tier,
                    AVG(output_tokens) AS avg_output,
                    AVG(
                        CASE
                            WHEN input_tokens > 0
                            THEN CAST(cached_tokens AS REAL) / input_tokens
                            ELSE 0.0
                        END
                    ) AS cache_hit_ratio,
                    COUNT(*) AS observations
                FROM routing_calls
                WHERE task_class = ?
                GROUP BY tier
                """,
                (task_class,),
            ).fetchall()
        return {
            str(row["tier"]): (
                float(row["avg_output"]) if row["avg_output"] is not None else None,
                float(row["cache_hit_ratio"] or 0.0),
                int(row["observations"] or 0),
            )
            for row in rows
        }

    def record_attempt(
        self,
        *,
        session_id: str,
        task_id: str,
        intent: str,
        tier: str,
        route_class: str,
        model: str,
        provider: str,
        success: bool,
        status_code: int | None,
        error_type: str,
        latency_s: float,
        cooldown_s: float,
    ) -> int:
        timestamp = time.time()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO routing_attempts (
                    session_id, task_id, timestamp, intent, tier, route_class,
                    model, provider, success, status_code, error_type,
                    latency_s, cooldown_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    task_id,
                    timestamp,
                    intent,
                    tier,
                    route_class,
                    model,
                    provider,
                    int(success),
                    status_code,
                    error_type,
                    latency_s,
                    cooldown_s,
                ),
            )
        return int(cursor.lastrowid)

    def list_attempts(self, session_id: str) -> list[RouteAttempt]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM routing_attempts
                WHERE session_id = ?
                ORDER BY attempt_id
                """,
                (session_id,),
            ).fetchall()
        return [RouteAttempt(**dict(row)) for row in rows]







