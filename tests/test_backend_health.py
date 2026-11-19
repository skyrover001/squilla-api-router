from squilla_api_router.backend_health import CircuitBreaker


def test_circuit_breaker_cooldown_and_reset() -> None:
    breaker = CircuitBreaker(cooldown_seconds=60, rate_limit_cooldown_seconds=15)
    assert breaker.record_failure("c1", status=503, now=100.0) == 60
    assert not breaker.available("c1", now=159.0)
    assert breaker.available("c1", now=160.0)
    assert breaker.record_failure("c1", status=429, now=200.0) == 15
    breaker.record_success("c1")
    assert "c1" not in breaker.snapshot()
