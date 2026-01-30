"""Confidence gating for LLM responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .exceptions import LowConfidenceError
from .providers.base import CompletionResult


@dataclass
class ConfidenceGate:
    """Gate that checks response confidence and optionally escalates.

    If confidence is below threshold:
      1. Calls escalation_callback (if set) — e.g. flag for human review.
      2. Returns fallback_response (if set).
      3. Otherwise raises LowConfidenceError.
    """

    min_threshold: float = 0.7
    escalation_callback: Callable[[CompletionResult], Awaitable[None]] | None = None
    fallback_response: str | None = None

    async def check(self, result: CompletionResult) -> CompletionResult:
        """Check confidence and either pass through, escalate, or reject."""
        if result.confidence is None or result.confidence >= self.min_threshold:
            return result

        # Low confidence path
        if self.escalation_callback:
            await self.escalation_callback(result)

        if self.fallback_response is not None:
            return CompletionResult(
                text=self.fallback_response,
                provider_used="confidence_gate_fallback",
                confidence=0.0,
                metadata={
                    "original_provider": result.provider_used,
                    "original_confidence": result.confidence,
                    "reason": "below_threshold",
                },
            )

        raise LowConfidenceError(result.confidence, self.min_threshold)
