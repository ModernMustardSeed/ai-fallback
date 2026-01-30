"""Adaptive timeout strategies for LLM calls."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AdaptiveTimeout:
    """Timeout that adapts based on prompt length and historical latency."""

    base_timeout: float = 30.0
    max_timeout: float = 120.0
    min_timeout: float = 5.0
    backoff_factor: float = 1.5
    prompt_length_factor: float = 0.01  # extra seconds per 100 chars
    _recent_latencies: list[float] = field(default_factory=list)
    _max_history: int = 20

    def get_timeout(self, prompt_length: int = 0, attempt: int = 1) -> float:
        """Calculate timeout for a request.

        Args:
            prompt_length: Character count of the prompt.
            attempt: Which attempt this is (1-indexed). Increases timeout via backoff.
        """
        base = self._adaptive_base()
        # Add time for longer prompts
        length_bonus = (prompt_length / 100) * self.prompt_length_factor
        # Backoff for retries
        retry_multiplier = self.backoff_factor ** (attempt - 1)
        timeout = (base + length_bonus) * retry_multiplier
        return max(self.min_timeout, min(timeout, self.max_timeout))

    def record_latency(self, latency_seconds: float) -> None:
        """Record an observed latency to inform future timeouts."""
        self._recent_latencies.append(latency_seconds)
        if len(self._recent_latencies) > self._max_history:
            self._recent_latencies = self._recent_latencies[-self._max_history :]

    def _adaptive_base(self) -> float:
        """Use p95 of recent latencies if available, else base_timeout."""
        if len(self._recent_latencies) < 3:
            return self.base_timeout
        sorted_lats = sorted(self._recent_latencies)
        p95_idx = int(len(sorted_lats) * 0.95)
        p95 = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]
        # Use p95 * 1.5 as the adaptive base, but don't go below base_timeout
        return max(self.base_timeout, p95 * 1.5)

    def reset(self) -> None:
        self._recent_latencies.clear()
