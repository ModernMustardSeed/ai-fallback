# ai-fallback

Production-grade LLM reliability patterns. Stop losing requests to provider outages, timeouts, and budget overruns.

## The Problem

Your AI agent calls Claude. Claude is down. Your user sees an error. You lose revenue. You lose trust.

**ai-fallback** gives you:
- **Fallback chains** — Claude fails? OpenAI picks up. OpenAI fails? Local model catches it.
- **Circuit breakers** — Stop hammering a dead provider. Auto-recover when it's back.
- **Cost protection** — Per-minute spend limits and total budget caps.
- **Adaptive timeouts** — Timeouts that learn from your traffic patterns.
- **Confidence gating** — Low-confidence responses get flagged for human review.
- **Partial recovery** — Salvage usable content from truncated streaming responses.
- **Structured observability** — JSONL logs with latency, cost, and retry tracking.

## Quick Start

```bash
pip install ai-fallback
```

```python
import asyncio
from ai_fallback import FallbackChain, ClaudeProvider, OpenAIProvider, LocalProvider

chain = FallbackChain([
    ClaudeProvider(api_key="sk-ant-...", model="claude-sonnet-4-20250514"),
    OpenAIProvider(api_key="sk-...", model="gpt-4o"),
    LocalProvider(model="llama3"),
])

result = asyncio.run(chain.complete("Summarize this document..."))
print(result.text)
print(f"Used: {result.provider_used}, Cost: ${result.cost_incurred:.4f}")
```

## Cost Protection

```python
from ai_fallback import FallbackChain, FallbackChainConfig, CircuitBreaker

chain = FallbackChain(
    providers=[claude, openai, local],
    config=FallbackChainConfig(budget_limit=1.00),  # $1 total cap
)

# Per-provider cost rate limiting
chain.set_circuit_breaker("claude", CircuitBreaker(
    provider_name="claude",
    cost_per_minute_limit=0.50,  # $0.50/min max
))
```

## Confidence Gating

```python
from ai_fallback import ConfidenceGate

async def flag_for_review(result):
    await send_to_slack(f"Low confidence response: {result.confidence}")

gate = ConfidenceGate(
    min_threshold=0.7,
    escalation_callback=flag_for_review,
    fallback_response="This has been escalated to a human.",
)
```

## Philosophy

1. **No provider SDKs** — Raw httpx calls. No dependency bloat.
2. **Async-first** — Built for production async workloads.
3. **Observable by default** — Every call is logged with latency, cost, and provider.
4. **Fail gracefully** — Every failure mode has a recovery path.
5. **Cost-aware** — Your AI agent shouldn't drain your bank account.

## Production Checklist

- [ ] Set `budget_limit` to prevent runaway costs
- [ ] Configure circuit breakers with `cost_per_minute_limit`
- [ ] Point `LLMObserver` at a persistent log file
- [ ] Set up a local fallback provider (Ollama) for when all APIs are down
- [ ] Configure confidence gating for high-stakes responses
- [ ] Test your fallback chain with `examples/basic_fallback.py`

## Cost Examples (approximate)

| Provider | Model | Input (1K tokens) | Output (1K tokens) |
|----------|-------|--------------------|---------------------|
| Anthropic | Claude Sonnet 4 | $0.003 | $0.015 |
| OpenAI | GPT-4o | $0.0025 | $0.010 |
| Local | Llama 3 | $0.000 | $0.000 |

With a 3-provider chain and $1.00 budget, you get ~60 Claude calls or ~100 GPT-4o calls before the budget cap kicks in.

## License

MIT
