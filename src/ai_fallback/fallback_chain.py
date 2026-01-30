"""FallbackChain — the main orchestrator for reliable LLM calls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .confidence_gate import ConfidenceGate
from .exceptions import (
    AIFallbackError,
    BudgetExceededError,
    ProviderError,
)
from .observability import LLMObserver
from .partial_recovery import PartialRecovery
from .providers.base import BaseProvider, CompletionResult
from .timeout_strategies import AdaptiveTimeout


@dataclass
class FallbackChainConfig:
    """Configuration for the fallback chain."""

    budget_limit: float | None = None  # total $ limit
    enable_partial_recovery: bool = True
    enable_confidence_gate: bool = False
    confidence_threshold: float = 0.7


class FallbackChain:
    """Orchestrates multiple LLM providers with reliability patterns.

    Features:
      - Ordered provider fallback
      - Per-provider circuit breakers
      - Adaptive timeouts
      - Cost budget enforcement
      - Partial response recovery
      - Confidence gating
      - Structured observability
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        *,
        config: FallbackChainConfig | None = None,
        timeout: AdaptiveTimeout | None = None,
        confidence_gate: ConfidenceGate | None = None,
        partial_recovery: PartialRecovery | None = None,
        observer: LLMObserver | None = None,
    ):
        self.providers = providers
        self.config = config or FallbackChainConfig()
        self.timeout = timeout or AdaptiveTimeout()
        self.confidence_gate = confidence_gate
        self.partial_recovery = partial_recovery or PartialRecovery()
        self.observer = observer or LLMObserver()
        self._total_cost: float = 0.0

        # Per-provider circuit breakers
        self._breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(provider_name=p.name) for p in providers
        }

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def set_circuit_breaker(self, provider_name: str, breaker: CircuitBreaker) -> None:
        self._breakers[provider_name] = breaker

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> CompletionResult:
        """Try each provider in order until one succeeds."""
        self._check_budget()
        errors: list[Exception] = []

        for i, provider in enumerate(self.providers):
            breaker = self._breakers.get(provider.name)

            # Check circuit breaker
            if breaker:
                try:
                    breaker.check()
                except CircuitOpenError as e:
                    self.observer.log_circuit_open(provider.name, str(e))
                    if i + 1 < len(self.providers):
                        self.observer.log_fallback(
                            provider.name,
                            self.providers[i + 1].name,
                            "circuit_open",
                        )
                    errors.append(e)
                    continue

            # Attempt the call
            for attempt in range(1, provider.config.max_retries + 1):
                self.observer.log_attempt(provider.name, prompt, attempt=attempt)
                timeout_s = self.timeout.get_timeout(
                    prompt_length=len(prompt), attempt=attempt
                )
                t0 = time.monotonic()

                try:
                    result = await asyncio.wait_for(
                        provider.complete(
                            prompt,
                            system=system,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        ),
                        timeout=timeout_s,
                    )
                    elapsed = (time.monotonic() - t0) * 1000
                    result.latency_ms = elapsed
                    result.attempts = attempt

                    # Record metrics
                    self.timeout.record_latency(elapsed / 1000)
                    self._total_cost += result.cost_incurred
                    provider.record_call(result)
                    if breaker:
                        breaker.record_success()
                        breaker.record_cost(result.cost_incurred)

                    # Confidence gate
                    if self.confidence_gate and self.config.enable_confidence_gate:
                        result = await self.confidence_gate.check(result)

                    self.observer.log_success(result)
                    return result

                except asyncio.TimeoutError:
                    elapsed = (time.monotonic() - t0) * 1000
                    err_msg = f"Timeout after {elapsed:.0f}ms"
                    self.observer.log_failure(provider.name, err_msg, attempt=attempt)
                    provider.record_failure()
                    if breaker:
                        breaker.record_failure()
                    errors.append(
                        ProviderError(provider.name, err_msg)
                    )

                except Exception as e:
                    elapsed = (time.monotonic() - t0) * 1000
                    self.observer.log_failure(provider.name, str(e), attempt=attempt)
                    provider.record_failure()
                    if breaker:
                        breaker.record_failure()
                    errors.append(e)

            # All retries exhausted for this provider — fall back
            if i + 1 < len(self.providers):
                self.observer.log_fallback(
                    provider.name,
                    self.providers[i + 1].name,
                    "retries_exhausted",
                )

        raise AIFallbackError(
            f"All {len(self.providers)} providers failed. Errors: {errors}"
        )

    def _check_budget(self) -> None:
        if self.config.budget_limit is not None:
            if self._total_cost >= self.config.budget_limit:
                self.observer.log_budget_warning(self._total_cost, self.config.budget_limit)
                raise BudgetExceededError(self.config.budget_limit, self._total_cost)
