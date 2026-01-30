"""Example: cost-protected agent with budget limits and circuit breakers."""

import asyncio
from ai_fallback import (
    FallbackChain,
    FallbackChainConfig,
    CircuitBreaker,
    BudgetExceededError,
    CompletionResult,
)
from ai_fallback.providers.base import BaseProvider, ProviderConfig


class FakeProvider(BaseProvider):
    def __init__(self, name, cost=0.01):
        super().__init__(ProviderConfig(name=name, max_retries=1))
        self._cost = cost

    async def complete(self, prompt, **kwargs):
        return CompletionResult(
            text=f"Response from {self.name}",
            provider_used=self.name,
            cost_incurred=self._cost,
        )

    async def stream(self, prompt, **kwargs):
        yield f"Response from {self.name}"


async def main():
    chain = FallbackChain(
        providers=[
            FakeProvider("gpt-4o", cost=0.03),
            FakeProvider("claude-haiku", cost=0.005),
        ],
        config=FallbackChainConfig(budget_limit=0.10),
    )

    # Set a cost-rate circuit breaker on the expensive provider
    chain.set_circuit_breaker(
        "gpt-4o",
        CircuitBreaker(provider_name="gpt-4o", cost_per_minute_limit=0.05),
    )

    for i in range(10):
        try:
            result = await chain.complete(f"Question {i}")
            print(f"[{i}] {result.provider_used}: ${result.cost_incurred:.4f}")
        except BudgetExceededError as e:
            print(f"[{i}] Budget exceeded: {e}")
            break

    print(f"\nTotal spend: ${chain.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
