"""Tests for CircuitBreaker."""

import time
import pytest
from ai_fallback import CircuitBreaker, CircuitOpenError
from ai_fallback.circuit_breaker import CircuitState


def test_starts_closed():
    cb = CircuitBreaker(provider_name="test")
    assert cb.state == CircuitState.CLOSED
    cb.check()  # should not raise


def test_opens_after_threshold():
    cb = CircuitBreaker(provider_name="test", failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        cb.check()


def test_success_resets_count():
    cb = CircuitBreaker(provider_name="test", failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    # Should still be closed — success reset the count
    assert cb.state == CircuitState.CLOSED


def test_half_open_after_recovery(monkeypatch):
    cb = CircuitBreaker(provider_name="test", failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    time.sleep(0.02)
    assert cb.state == CircuitState.HALF_OPEN


def test_cost_rate_limit():
    cb = CircuitBreaker(provider_name="test", cost_per_minute_limit=0.05)
    for _ in range(10):
        cb.record_cost(0.01)  # 0.10 total in <1 min
    with pytest.raises(CircuitOpenError, match="cost rate"):
        cb.check()


def test_reset():
    cb = CircuitBreaker(provider_name="test", failure_threshold=1)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED
