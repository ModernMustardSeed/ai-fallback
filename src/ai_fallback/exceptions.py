"""Custom exception hierarchy for ai-fallback."""


class AIFallbackError(Exception):
    """Base exception for all ai-fallback errors."""


class ProviderError(AIFallbackError):
    """A provider failed to return a valid response."""

    def __init__(self, provider_name: str, message: str, status_code: int | None = None):
        self.provider_name = provider_name
        self.status_code = status_code
        super().__init__(f"[{provider_name}] {message}")


class TimeoutError(AIFallbackError):
    """A provider call exceeded its timeout."""

    def __init__(self, provider_name: str, timeout_seconds: float):
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"[{provider_name}] Timed out after {timeout_seconds:.1f}s")


class CircuitOpenError(AIFallbackError):
    """The circuit breaker is open for a provider — calls are blocked."""

    def __init__(self, provider_name: str, reason: str = "too many recent failures"):
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] Circuit open: {reason}")


class BudgetExceededError(AIFallbackError):
    """Cost budget has been exceeded."""

    def __init__(self, budget_limit: float, current_spend: float):
        self.budget_limit = budget_limit
        self.current_spend = current_spend
        super().__init__(
            f"Budget exceeded: ${current_spend:.4f} spent, limit is ${budget_limit:.4f}"
        )


class LowConfidenceError(AIFallbackError):
    """The response confidence score is below the required threshold."""

    def __init__(self, score: float, threshold: float):
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"Confidence {score:.2f} below threshold {threshold:.2f}"
        )
