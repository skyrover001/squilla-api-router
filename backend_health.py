"""Process-local circuit breaker for model backend attempts."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


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
    ) -> None:
        self.enabled = enabled
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.rate_limit_cooldown_seconds = max(0.0, rate_limit_cooldown_seconds)
        self._state: dict[str, BackendHealth] = {}
        self._lock = threading.Lock()

    def available(self, key: str, now: float | None = None) -> bool:
        if not self.enabled:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            state = self._state.get(key)
            return state is None or state.cooldown_until <= now

    def record_success(self, key: str) -> None:
        if not self.enabled:
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
        now = time.monotonic() if now is None else now
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
        with self._lock:
            return dict(self._state)

