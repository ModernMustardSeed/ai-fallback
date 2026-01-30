"""Basic fallback chain example with mock providers."""

import asyncio
from ai_fallback import FallbackChain, CompletionResult
from ai_fallback.providers.base import BaseProvider, ProviderConfig


class DemoProvider(BaseProvider):
    """Simple demo provider that returns canned responses."""

    def __init__(self, name: str, response: str, fail: bool = False):
        super().__init__(ProviderConfig(name=name, max_retries=1))
        self._response = response
        self._fail = fail

    async def complete(self, prompt, **kwargs):
        if self._fail:
            raise RuntimeError(f"{self.name} is down!")
        return CompletionResult(
            text=self._response,
            provider_used=self.name,
            cost_incurred=0.001,
        )

    async def stream(self, prompt, **kwargs):
        for word in self._response.split():
            yield word + " "


async def main():
    chain = FallbackChain([
        DemoProvider("primary", "Answer from primary", fail=True),
        DemoProvider("secondary", "Answer from secondary"),
        DemoProvider("local", "Answer from local fallback"),
    ])

    result = await chain.complete("What is the meaning of life?")
    print(f"Provider used: {result.provider_used}")
    print(f"Response: {result.text}")
    print(f"Total cost: ${chain.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
