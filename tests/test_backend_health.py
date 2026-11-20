from pathlib import Path
from uuid import uuid4

from squilla_api_router.backend_health import CircuitBreaker


def test_circuit_breaker_cooldown_and_reset() -> None:
    breaker = CircuitBreaker(cooldown_seconds=60, rate_limit_cooldown_seconds=15)
    assert breaker.record_failure("c1", status=503, now=100.0) == 60
    assert not breaker.available("c1", now=159.0)
    assert breaker.available("c1", now=160.0)
    assert breaker.record_failure("c1", status=429, now=200.0) == 15
    breaker.record_success("c1")
    assert "c1" not in breaker.snapshot()


def test_circuit_breaker_state_is_shared_across_instances() -> None:
    state_path = Path(__file__).with_name(f".circuit-{uuid4().hex}.db")
    try:
        first = CircuitBreaker(
            cooldown_seconds=60,
            state_path=state_path,
        )
        assert first.record_failure("provider|model", status=503, now=100.0) == 60

        second = CircuitBreaker(
            cooldown_seconds=60,
            state_path=state_path,
        )
        assert not second.available("provider|model", now=101.0)
        assert second.available("provider|model", now=160.0)

        second.record_success("provider|model")
        third = CircuitBreaker(
            cooldown_seconds=60,
            state_path=state_path,
        )
        assert third.available("provider|model", now=101.0)
        assert third.snapshot() == {}
    finally:
        for suffix in ("", "-shm", "-wal"):
            Path(str(state_path) + suffix).unlink(missing_ok=True)
