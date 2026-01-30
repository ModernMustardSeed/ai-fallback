"""Abstract base provider for LLM APIs."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    name: str
    timeout_seconds: float = 30.0
    max_retries: int = 2
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    api_key: str = ""
    base_url: str = ""
    model: str = ""


@dataclass
class CompletionResult:
    """Result of a completion call."""

    text: str
    provider_used: str
    model: str = ""
    attempts: int = 1
    latency_ms: float = 0.0
    cost_incurred: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    confidence: float | None = None
    truncated: bool = False
    metadata: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract LLM provider."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.total_cost: float = 0.0
        self.total_calls: int = 0
        self.total_failures: int = 0

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> CompletionResult:
        """Send a completion request and return the full result."""

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return (
            input_tokens * self.config.cost_per_input_token
            + output_tokens * self.config.cost_per_output_token
        )

    def record_call(self, result: CompletionResult) -> None:
        """Track cost and usage after a call."""
        self.total_cost += result.cost_incurred
        self.total_calls += 1

    def record_failure(self) -> None:
        self.total_failures += 1
        self.total_calls += 1
