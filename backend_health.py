"""Circuit breaker for model backend attempts.

When ``state_path`` is configured the breaker state is shared through SQLite,
so multiple router workers and process restarts observe the same cooldown.
Without ``state_path`` the original process-local behaviour is retained.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendHealth:
    consecutive_failures: int
    cooldown_until: float
    last_status: int | None
    last_error: str


class CircuitBreaker:
    def __init__(
        self,
        *,
        enabled: bool = True,
        failure_threshold: int = 1,
        cooldown_seconds: float = 60.0,
        rate_limit_cooldown_seconds: float = 15.0,
        state_path: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.rate_limit_cooldown_seconds = max(0.0, rate_limit_cooldown_seconds)
        self.state_path = Path(state_path) if state_path else None
        self._state: dict[str, BackendHealth] = {}
        self._lock = threading.Lock()
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_store()

    def _connect(self) -> sqlite3.Connection:
        if self.state_path is None:
            raise RuntimeError("persistent circuit state is not configured")
        connection = sqlite3.connect(self.state_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize_store(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS backend_circuit_state (
                    backend_key TEXT PRIMARY KEY,
                    consecutive_failures INTEGER NOT NULL,
                    cooldown_until REAL NOT NULL,
                    last_status INTEGER,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _row_health(row: sqlite3.Row) -> BackendHealth:
        return BackendHealth(
            consecutive_failures=int(row["consecutive_failures"]),
            cooldown_until=float(row["cooldown_until"]),
            last_status=(
                int(row["last_status"]) if row["last_status"] is not None else None
            ),
            last_error=str(row["last_error"] or ""),
        )

    def available(self, key: str, now: float | None = None) -> bool:
        if not self.enabled:
            return True
        if self.state_path is not None:
            now = time.time() if now is None else now
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT consecutive_failures, cooldown_until, last_status,
                           last_error
                    FROM backend_circuit_state
                    WHERE backend_key = ?
                    """,
                    (key,),
                ).fetchone()
            state = self._row_health(row) if row is not None else None
            return state is None or state.cooldown_until <= now
        now = time.monotonic() if now is None else now
        with self._lock:
            state = self._state.get(key)
            return state is None or state.cooldown_until <= now

    def record_success(self, key: str) -> None:
        if not self.enabled:
            return
        if self.state_path is not None:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "DELETE FROM backend_circuit_state WHERE backend_key = ?",
                    (key,),
                )
            return
        with self._lock:
            self._state.pop(key, None)

    def record_failure(
        self,
        key: str,
        *,
        status: int | None = None,
        error: str = "",
        now: float | None = None,
    ) -> float:
        if not self.enabled:
            return 0.0
        if now is None:
            now = time.time() if self.state_path is not None else time.monotonic()
        if self.state_path is not None:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT consecutive_failures, cooldown_until, last_status,
                           last_error
                    FROM backend_circuit_state
                    WHERE backend_key = ?
                    """,
                    (key,),
                ).fetchone()
                previous = self._row_health(row) if row is not None else None
                failures = (
                    previous.consecutive_failures if previous is not None else 0
                ) + 1
                cooldown = (
                    self.rate_limit_cooldown_seconds
                    if status == 429
                    else self.cooldown_seconds
                )
                until = now + cooldown if failures >= self.failure_threshold else 0.0
                connection.execute(
                    """
                    INSERT INTO backend_circuit_state (
                        backend_key, consecutive_failures, cooldown_until,
                        last_status, last_error, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(backend_key) DO UPDATE SET
                        consecutive_failures = excluded.consecutive_failures,
                        cooldown_until = excluded.cooldown_until,
                        last_status = excluded.last_status,
                        last_error = excluded.last_error,
                        updated_at = excluded.updated_at
                    """,
                    (key, failures, until, status, error[:1000], now),
                )
            return cooldown if until > now else 0.0
        with self._lock:
            previous = self._state.get(key)
            failures = (previous.consecutive_failures if previous else 0) + 1
            cooldown = (
                self.rate_limit_cooldown_seconds
                if status == 429
                else self.cooldown_seconds
            )
            until = now + cooldown if failures >= self.failure_threshold else 0.0
            self._state[key] = BackendHealth(
                consecutive_failures=failures,
                cooldown_until=until,
                last_status=status,
                last_error=error,
            )
            return cooldown if until > now else 0.0

    def snapshot(self) -> dict[str, BackendHealth]:
        if self.state_path is not None:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT backend_key, consecutive_failures, cooldown_until,
                           last_status, last_error
                    FROM backend_circuit_state
                    """
                ).fetchall()
            return {
                str(row["backend_key"]): self._row_health(row) for row in rows
            }
        with self._lock:
            return dict(self._state)

