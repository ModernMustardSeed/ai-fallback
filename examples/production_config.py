"""Example: production configuration with all reliability patterns."""

import asyncio
import os
from ai_fallback import (
    FallbackChain,
    FallbackChainConfig,
    AdaptiveTimeout,
    CircuitBreaker,
    ConfidenceGate,
    LLMObserver,
    PartialRecovery,
    ClaudeProvider,
    OpenAIProvider,
    LocalProvider,
)


def build_production_chain() -> FallbackChain:
    """Build a production-ready fallback chain.

    Requires API keys in environment variables:
      - ANTHROPIC_API_KEY
      - OPENAI_API_KEY
    """
    claude = ClaudeProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-4-20250514",
        cost_per_input_token=3e-6,
        cost_per_output_token=15e-6,
    )
    openai = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="gpt-4o",
        cost_per_input_token=2.5e-6,
        cost_per_output_token=10e-6,
    )
    local = LocalProvider(model="llama3")

    async def escalate(result):
        print(f"LOW CONFIDENCE: {result.confidence} — flagging for review")

    chain = FallbackChain(
        providers=[claude, openai, local],
        config=FallbackChainConfig(
            budget_limit=1.00,
            enable_confidence_gate=True,
            confidence_threshold=0.6,
        ),
        timeout=AdaptiveTimeout(base_timeout=30.0, max_timeout=90.0),
        confidence_gate=ConfidenceGate(
            min_threshold=0.6,
            escalation_callback=escalate,
            fallback_response="Flagged for human review.",
        ),
        partial_recovery=PartialRecovery(min_usable_length=100),
        observer=LLMObserver(log_file="llm_calls.jsonl"),
    )

    # Custom circuit breakers per provider
    chain.set_circuit_breaker(
        "claude",
        CircuitBreaker(provider_name="claude", failure_threshold=3, cost_per_minute_limit=0.50),
    )
    chain.set_circuit_breaker(
        "openai",
        CircuitBreaker(provider_name="openai", failure_threshold=5, cost_per_minute_limit=0.30),
    )

    return chain


async def main():
    chain = build_production_chain()
    print("Production chain configured with 3 providers.")
    print("Claude -> OpenAI -> Local (Ollama)")
    print(f"Budget limit: ${chain.config.budget_limit}")
    print("Run with valid API keys to test live calls.")


if __name__ == "__main__":
    asyncio.run(main())
