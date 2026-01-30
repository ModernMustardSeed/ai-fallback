"""ai-fallback: Production-grade LLM reliability patterns."""

from .circuit_breaker import CircuitBreaker, CircuitState
from .confidence_gate import ConfidenceGate
from .exceptions import (
    AIFallbackError,
    BudgetExceededError,
    CircuitOpenError,
    LowConfidenceError,
    ProviderError,
    TimeoutError,
)
from .fallback_chain import FallbackChain, FallbackChainConfig
from .observability import LLMObserver
from .partial_recovery import PartialRecovery
from .providers.base import BaseProvider, CompletionResult, ProviderConfig
from .providers.claude import ClaudeProvider
from .providers.local import LocalProvider
from .providers.openai import OpenAIProvider
from .timeout_strategies import AdaptiveTimeout

__all__ = [
    "AdaptiveTimeout",
    "AIFallbackError",
    "BaseProvider",
    "BudgetExceededError",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "ClaudeProvider",
    "CompletionResult",
    "ConfidenceGate",
    "FallbackChain",
    "FallbackChainConfig",
    "LLMObserver",
    "LocalProvider",
    "LowConfidenceError",
    "OpenAIProvider",
    "PartialRecovery",
    "ProviderConfig",
    "ProviderError",
    "TimeoutError",
]

__version__ = "0.1.0"
