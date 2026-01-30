"""Example: confidence gating with human escalation."""

import asyncio
from ai_fallback import (
    FallbackChain,
    FallbackChainConfig,
    ConfidenceGate,
    CompletionResult,
)
from ai_fallback.providers.base import BaseProvider, ProviderConfig


class UncertainProvider(BaseProvider):
    def __init__(self):
        super().__init__(ProviderConfig(name="uncertain-llm", max_retries=1))

    async def complete(self, prompt, **kwargs):
        return CompletionResult(
            text="I think the answer might be 42, but I'm not sure.",
            provider_used=self.name,
            confidence=0.35,
            cost_incurred=0.01,
        )

    async def stream(self, prompt, **kwargs):
        yield "I think the answer might be 42."


async def main():
    escalation_queue: list[CompletionResult] = []

    async def escalate_to_human(result: CompletionResult):
        print(f"  >> ESCALATED to human review: confidence={result.confidence}")
        escalation_queue.append(result)

    gate = ConfidenceGate(
        min_threshold=0.7,
        escalation_callback=escalate_to_human,
        fallback_response="This question has been flagged for human review.",
    )

    chain = FallbackChain(
        providers=[UncertainProvider()],
        config=FallbackChainConfig(enable_confidence_gate=True),
        confidence_gate=gate,
    )

    result = await chain.complete("What is the airspeed velocity of an unladen swallow?")
    print(f"Response: {result.text}")
    print(f"Provider: {result.provider_used}")
    print(f"Escalations queued: {len(escalation_queue)}")


if __name__ == "__main__":
    asyncio.run(main())
