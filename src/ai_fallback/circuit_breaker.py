"""Circuit breaker for LLM providers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import CircuitOpenError


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker with error-rate and cost-rate thresholds.

    - CLOSED: normal operation.
    - OPEN: calls blocked after threshold breached.
    - HALF_OPEN: one trial call allowed after recovery_timeout.
    """

    provider_name: str = ""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds before trying half-open
    cost_per_minute_limit: float | None = None  # max $/min spend

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _cost_window: list[tuple[float, float]] = field(default_factory=list, init=False)  # (timestamp, cost)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def check(self) -> None:
        """Raise CircuitOpenError if circuit is open."""
        state = self.state
        if state == CircuitState.OPEN:
            self._raise_open("too many recent failures")
        # Check cost rate
        if self.cost_per_minute_limit is not None:
            rate = self._cost_rate()
            if rate >= self.cost_per_minute_limit:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.time()
                self._raise_open(f"cost rate ${rate:.4f}/min exceeds limit ${self.cost_per_minute_limit:.4f}/min")

    def record_success(self) -> None:
        """Reset failure count on success."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failure; open circuit if threshold reached."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def record_cost(self, cost: float) -> None:
        """Track cost for rate limiting."""
        self._cost_window.append((time.time(), cost))
        # Prune entries older than 60s
        cutoff = time.time() - 60.0
        self._cost_window = [(t, c) for t, c in self._cost_window if t >= cutoff]

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._cost_window.clear()

    def _cost_rate(self) -> float:
        cutoff = time.time() - 60.0
        self._cost_window = [(t, c) for t, c in self._cost_window if t >= cutoff]
        return sum(c for _, c in self._cost_window)

    def _raise_open(self, reason: str) -> None:
        raise CircuitOpenError(self.provider_name, reason)
