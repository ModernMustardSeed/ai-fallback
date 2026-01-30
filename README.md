# ai-fallback

[![CI](https://github.com/ModernMustardSeed/ai-fallback/actions/workflows/ci.yml/badge.svg)](https://github.com/ModernMustardSeed/ai-fallback/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-grade LLM reliability patterns. Stop losing requests to provider outages, timeouts, and budget overruns.

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                    FallbackChain                        │
                          │                                                         │
  prompt ──────►  Budget Check ──► Circuit Breaker ──► Adaptive Timeout ──► Provider │
                          │              │                    │                │     │
                          │              │ open               │ timeout        │     │
                          │              ▼                    ▼                ▼     │
                          │         next provider ◄──── next provider    Confidence  │
                          │              │                                  Gate     │
                          │              │                                   │      │
                          │              ▼                                   ▼      │
                          │     ... until chain exhausted           Result / Escalate│
                          └─────────────────────────────────────────────────────────┘
```

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

> **Deep dive:** See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical design, failure scenario analysis, and cost modeling.

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

## Adaptive Timeouts

```python
from ai_fallback import AdaptiveTimeout

timeout = AdaptiveTimeout(
    base_timeout=30.0,      # starting point
    max_timeout=90.0,       # hard ceiling
    backoff_factor=1.5,     # multiplier per retry
    prompt_length_factor=0.01,  # extra seconds per 100 chars
)
# Learns from observed latencies — uses p95 to set adaptive base
```

## Observability

Every call produces structured JSONL events:

```json
{"ts": 1706640000.0, "event": "attempt", "provider": "claude", "attempt": 1, "prompt_preview": "Summarize this..."}
{"ts": 1706640001.2, "event": "success", "provider": "claude", "latency_ms": 1200.0, "cost": 0.0045, "attempts": 1}
{"ts": 1706640002.0, "event": "failure", "provider": "claude", "error": "Timeout after 30000ms", "attempt": 1}
{"ts": 1706640002.0, "event": "fallback", "from_provider": "claude", "to_provider": "openai", "reason": "retries_exhausted"}
{"ts": 1706640003.0, "event": "circuit_open", "provider": "claude", "reason": "too many recent failures"}
```

## Failure Scenarios

| Scenario | What Happens |
|----------|-------------|
| Provider returns 500 | Retries with backoff, then falls back to next provider |
| Provider times out | Adaptive timeout triggers, falls back |
| Provider down for minutes | Circuit breaker opens, skips provider entirely, auto-recovers via half-open |
| Cost spike | Per-minute circuit breaker trips, budget cap prevents overspend |
| Low-confidence response | Escalation callback fires, fallback response returned |
| Streaming response truncated | Partial recovery trims to last complete sentence |
| All providers fail | `AIFallbackError` raised with full error chain |
| Budget exhausted | `BudgetExceededError` before any API call is made |

## Philosophy

1. **No provider SDKs** — Raw httpx calls. No dependency bloat. Two runtime deps: `httpx` + `pydantic`.
2. **Async-first** — LLM calls are I/O-bound. Async is the only sane default.
3. **Observable by default** — Every call is logged with latency, cost, and provider.
4. **Fail gracefully** — Every failure mode has a defined recovery path.
5. **Cost-aware** — Your AI agent shouldn't drain your bank account at 3am.

## Project Structure

```
src/ai_fallback/
├── __init__.py              # Public API
├── fallback_chain.py        # Core orchestrator
├── circuit_breaker.py       # Error-rate + cost-rate circuit breaker
├── timeout_strategies.py    # Adaptive p95-based timeouts
├── confidence_gate.py       # Threshold gating with escalation
├── partial_recovery.py      # Truncated response salvaging
├── observability.py         # Structured JSONL logging
├── exceptions.py            # Exception hierarchy
└── providers/
    ├── base.py              # Abstract provider + CompletionResult
    ├── claude.py            # Anthropic Messages API
    ├── openai.py            # OpenAI Chat Completions API
    └── local.py             # Ollama-compatible local models
```

## Production Checklist

- [ ] Set `budget_limit` to prevent runaway costs
- [ ] Configure circuit breakers with `cost_per_minute_limit`
- [ ] Point `LLMObserver` at a persistent log file
- [ ] Set up a local fallback provider (Ollama) for when all APIs are down
- [ ] Configure confidence gating for high-stakes responses
- [ ] Test your fallback chain with `examples/basic_fallback.py`
- [ ] Monitor JSONL logs for latency trends and fallback frequency

## Examples

- [`basic_fallback.py`](examples/basic_fallback.py) — Minimal 3-provider chain
- [`cost_protected_agent.py`](examples/cost_protected_agent.py) — Budget limits + cost-rate circuit breakers
- [`human_escalation_flow.py`](examples/human_escalation_flow.py) — Confidence gating with escalation
- [`production_config.py`](examples/production_config.py) — Full production setup with all patterns

## License

MIT
