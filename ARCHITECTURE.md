# ai-fallback: Architecture Deep-Dive

**Production LLM reliability for systems that can't afford to be wrong or down.**

---

## Why This Exists

Every team building on LLMs hits the same wall around month three of production. The demo worked great. The pilot was fine. Then reality shows up:

- **Provider outages.** OpenAI has had multiple major incidents -- the November 2023 outage took the API down for hours, and partial degradations (elevated 500s, 2-3x latency) happen monthly. Anthropic rate limits hit hard under burst traffic. If your product has one provider, your uptime is their uptime.

- **Latency spikes.** LLM inference is not like calling a REST API. A request that normally takes 2 seconds can take 45 seconds during a provider's capacity crunch. Static timeouts either cut off legitimate long responses or let your users stare at spinners during incidents.

- **Cost runaways.** A retry loop against GPT-4 with no cost awareness can burn through a daily budget in minutes. I've seen teams discover $400 in unexpected charges from a single bad deployment that hammered retries. The bill doesn't care about your good intentions.

- **Confidence failures.** This is the one that doesn't show up in your monitoring until it's too late. An LLM returns a plausible-sounding answer with low confidence, your agentic system takes an action based on it, and now you're explaining to a customer why their data got processed incorrectly. For systems that take actions -- not just display text -- confidence matters.

- **Streaming truncation.** Connections drop. Providers hit max token limits mid-sentence. You get back 80% of a perfectly good response and throw it all away because your code only handles "complete or error."

`ai-fallback` is a single library that addresses all of these failure modes with a unified request pipeline. No sprawling microservices, no vendor SDKs, no framework lock-in. Just reliability patterns composed into a chain.

---

## Architecture Overview

### Request Flow

```
                          ai-fallback request pipeline
 ============================================================================

 Client
   |
   v
 FallbackChain.complete(prompt)
   |
   |--- [1] Budget Check -----> BudgetExceededError (if over limit)
   |
   |--- [2] For each provider (ordered):
   |      |
   |      |--- [2a] Circuit Breaker check
   |      |      |
   |      |      |-- OPEN -----> skip to next provider
   |      |      |-- HALF_OPEN -> allow one trial request
   |      |      |-- CLOSED ---> proceed
   |      |
   |      |--- [2b] Retry loop (up to max_retries):
   |      |      |
   |      |      |--- Adaptive Timeout calculation
   |      |      |      (p95 latency history + prompt length + backoff)
   |      |      |
   |      |      |--- asyncio.wait_for(provider.complete(), timeout)
   |      |      |      |
   |      |      |      |-- Success:
   |      |      |      |     |--- Record latency for timeout adaptation
   |      |      |      |     |--- Record cost to circuit breaker
   |      |      |      |     |--- breaker.record_success()
   |      |      |      |     |
   |      |      |      |     |--- [3] Confidence Gate (optional)
   |      |      |      |     |      |-- Above threshold --> return result
   |      |      |      |     |      |-- Below threshold:
   |      |      |      |     |      |     |-- escalation_callback()
   |      |      |      |     |      |     |-- return fallback_response
   |      |      |      |     |      |     |-- or raise LowConfidenceError
   |      |      |      |     |
   |      |      |      |     +--- Result returned to client
   |      |      |      |
   |      |      |      |-- Timeout/Error:
   |      |      |            |--- breaker.record_failure()
   |      |      |            |--- observer.log_failure()
   |      |      |            |--- retry or fall through
   |      |      |
   |      |      +--- (next retry attempt)
   |      |
   |      |--- Retries exhausted --> fall to next provider
   |      |    observer.log_fallback(from, to, reason)
   |
   +--- All providers exhausted --> AIFallbackError

 ============================================================================
 Observability (JSONL) emits structured events at every decision point.
```

### Module Dependency Graph

```
 fallback_chain.py
   |--- circuit_breaker.py
   |--- confidence_gate.py
   |--- partial_recovery.py
   |--- timeout_strategies.py
   |--- observability.py
   |--- providers/
   |      |--- base.py (BaseProvider, CompletionResult, ProviderConfig)
   |      |--- claude.py   (httpx -> Anthropic API)
   |      |--- openai.py   (httpx -> OpenAI API)
   |      |--- local.py    (httpx -> Ollama API)
   |--- exceptions.py
         |--- AIFallbackError
         |--- ProviderError
         |--- TimeoutError
         |--- CircuitOpenError
         |--- BudgetExceededError
         |--- LowConfidenceError
```

---

## Pattern Deep-Dives

### Circuit Breaker (`circuit_breaker.py`)

The circuit breaker is a state machine with three states:

```
                     failure_count >= threshold
            +--------+                  +--------+
            |        |  ─────────────>  |        |
            | CLOSED |                  |  OPEN  |
            |        |  <─────────────  |        |
            +--------+   recovery_timeout expires +--------+
                ^                            |
                |         +───────────+      |
                |         |           |      |
                +─────────| HALF_OPEN |<─────+
                 success  |           |  one trial allowed
                          +───────────+
                               |
                               +── failure ──> back to OPEN
```

**CLOSED**: Normal operation. Every failure increments a counter. When the counter hits `failure_threshold` (default 5), the circuit opens.

**OPEN**: All calls are blocked immediately -- no network request, no latency, no cost. After `recovery_timeout` seconds (default 60), the state transitions to HALF_OPEN on the next `check()` call.

**HALF_OPEN**: One trial request is allowed through. If it succeeds, the circuit closes and the failure counter resets to zero. If it fails, the circuit re-opens and the recovery timer restarts.

#### Why cost-rate limiting matters

Error-rate circuit breaking is table stakes. The less obvious problem is cost. Consider: your primary provider is responding successfully, but latency has spiked and your retry logic is generating 3x the normal request volume. Each request succeeds, so the error-rate breaker stays closed. Meanwhile, you're burning through budget at triple the expected rate.

The cost-rate breaker tracks spend over a 60-second sliding window:

```python
def record_cost(self, cost: float) -> None:
    self._cost_window.append((time.time(), cost))
    cutoff = time.time() - 60.0
    self._cost_window = [(t, c) for t, c in self._cost_window if t >= cutoff]
```

When the rolling cost rate exceeds `cost_per_minute_limit`, the circuit opens regardless of error rate. This catches the "succeeding expensively" failure mode that pure error-rate breakers miss.

The sliding window prunes entries older than 60 seconds on every `record_cost` and `_cost_rate` call. This keeps the window bounded without a background timer -- no threads, no event loops, just lazy cleanup on access.

---

### Adaptive Timeout (`timeout_strategies.py`)

#### Why static timeouts fail for LLMs

LLM latency is not normally distributed. It's bimodal: most requests complete in 1-5 seconds, but a meaningful percentage take 15-60+ seconds (long outputs, complex reasoning, provider load). A static 30-second timeout either:

1. Kills legitimate long requests during normal operation, or
2. Lets your users wait 30 seconds during an outage before you even start the fallback

Neither is acceptable.

#### p95 adaptation

The adaptive timeout maintains a rolling window of the last 20 observed latencies and computes the p95:

```python
def _adaptive_base(self) -> float:
    if len(self._recent_latencies) < 3:
        return self.base_timeout  # not enough data yet
    sorted_lats = sorted(self._recent_latencies)
    p95_idx = int(len(sorted_lats) * 0.95)
    p95 = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]
    return max(self.base_timeout, p95 * 1.5)
```

The logic: if 95% of your recent requests completed within X seconds, a reasonable timeout is 1.5X. This adapts in both directions -- tightens during fast periods (triggering faster fallback) and loosens during legitimately slower periods (avoiding false timeouts).

The `base_timeout` acts as a floor. We never go below it, even if recent latencies have been very fast. This prevents the timeout from collapsing to something unreasonable during a lucky streak.

#### Prompt-length awareness

Longer prompts produce longer responses, which take longer to generate. The timeout formula accounts for this:

```python
timeout = (adaptive_base + length_bonus) * retry_multiplier
```

Where `length_bonus = (prompt_length / 100) * 0.01`. It's a small factor, but it prevents a 10,000-character prompt from being held to the same timeout as a 50-character prompt.

#### Retry backoff

Each retry attempt multiplies the timeout by `backoff_factor` (default 1.5). First attempt gets the calculated timeout; second attempt gets 1.5x; third gets 2.25x. The rationale: if the first attempt timed out, the provider might be temporarily slow. Giving the retry more headroom avoids burning through all attempts at the same too-tight timeout.

The final value is always clamped between `min_timeout` (5s) and `max_timeout` (120s).

---

### Confidence Gate (`confidence_gate.py`)

#### The escalation pattern

Not every LLM response should be acted on. This is especially critical in agentic systems where the model's output triggers real-world side effects -- sending emails, modifying data, executing transactions.

The confidence gate sits after a successful provider response and checks `CompletionResult.confidence` against a threshold:

```
Result arrives (confidence: 0.45)
  |
  |--- 0.45 < 0.70 threshold
  |
  |--- [1] escalation_callback(result)  --> e.g., send to Slack, flag for human review
  |--- [2] return fallback_response     --> safe default ("I'm not sure, let me check")
  |--- [3] raise LowConfidenceError     --> if no fallback configured
```

The three-tier design is deliberate:

1. **Escalation callback** fires first and is async. This is where you notify a human, log to a review queue, or trigger a secondary verification. It runs even if a fallback response is returned -- the system serves the safe fallback while a human reviews the uncertain case.

2. **Fallback response** is the safe answer returned to the caller. The metadata preserves the original provider and confidence score for audit.

3. **Error** is the last resort when you'd rather fail loudly than serve uncertain content.

This matters because most systems either blindly trust model output or require synchronous human approval. The escalation pattern gives you the third option: serve a safe default immediately, review async, and use the review to improve your prompts or thresholds.

---

### Partial Recovery (`partial_recovery.py`)

#### When responses get truncated

Streaming responses can be interrupted by:
- Network disconnection mid-stream
- Provider hitting `max_tokens` limit mid-sentence
- Client-side timeout during a streaming response
- Provider-side errors partway through generation

The naive approach discards the entire partial response and retries from scratch. But if you've received 800 tokens of a 1000-token response, you've already paid for those tokens and the content is likely useful.

#### Sentence-boundary detection

The recovery strategy trims to the last complete sentence:

```python
def _trim_to_last_sentence(self, text: str) -> str:
    match = list(re.finditer(r'[.!?]\s', text))
    if match:
        return text[: match[-1].end()].strip()
    if text and text[-1] in ".!?":
        return text
    return text
```

The regex `[.!?]\s` finds sentence-ending punctuation followed by whitespace -- this avoids false positives on abbreviations like "Dr." or "U.S." which are typically not followed by whitespace and a new sentence in the same way.

Paragraph trimming (`rsplit("\n\n", 1)`) is the more aggressive option, useful when you need structurally complete output (e.g., markdown sections).

Both strategies enforce `min_usable_length` (default 50 chars). A recovered result is tagged with `truncated=True` and metadata `{"recovered": True}`, so downstream consumers know what they're working with.

---

### Observability (`observability.py`)

#### Why JSONL

JSONL (newline-delimited JSON) is the right format for LLM operational logs because:

1. **Append-only.** Each event is one line. No file corruption from incomplete writes. No parsing the entire file to add an entry.
2. **Streamable.** `tail -f` works. Pipe to `jq` for ad-hoc queries. Ship to any log aggregator that accepts structured logs.
3. **Schema-flexible.** Different event types carry different fields. No need to pre-declare columns.
4. **Small.** No XML overhead, no array wrappers. Just data.

#### What gets logged

Every decision point in the pipeline emits a structured event:

| Event | Fields | Why It Matters |
|-------|--------|----------------|
| `attempt` | provider, attempt #, prompt preview (120 chars) | Request tracing, debugging retry storms |
| `success` | provider, model, latency_ms, cost, token counts, truncated | Cost tracking, latency SLOs, model performance |
| `failure` | provider, error message, attempt # | Error rates, categorizing failure modes |
| `circuit_open` | provider, reason | Alerting on provider degradation |
| `fallback` | from_provider, to_provider, reason | Understanding fallback frequency and triggers |
| `budget_warning` | current_spend, limit | Cost control, billing alerts |

The observer dual-writes: to a JSONL file for persistence and to an in-memory `events` list for programmatic access. The `_stream` parameter allows real-time output to stdout or any IO stream for debugging.

Prompt content is truncated to 120 characters in logs. Full prompts in logs are a security and privacy liability.

---

## Design Decisions

### Why no provider SDKs -- httpx only

The `anthropic` and `openai` Python SDKs are convenient for prototyping. They're wrong for a reliability library.

1. **Transitive dependencies.** The OpenAI SDK pulls in `httpx`, `pydantic`, `distro`, `sniffio`, `anyio`, `typing-extensions`, and more. The Anthropic SDK has a similar tree. Each dependency is an attack surface and an upgrade risk. `ai-fallback` depends on `httpx` and `pydantic` -- that's it.

2. **Vendor lock-in at the abstraction layer.** SDKs model their vendor's API surface. When you build your reliability layer on top of SDK objects (`openai.ChatCompletion`, `anthropic.Message`), your fallback logic is coupled to vendor-specific types. Raw httpx means every provider returns the same `CompletionResult` dataclass.

3. **Retry interference.** Both SDKs implement their own retry logic. When you're building a reliability layer with custom retry, circuit breaking, and timeout strategies, having the SDK also retrying underneath is a debugging nightmare. With raw httpx, we control every request.

4. **Version pinning pain.** When `openai` ships a breaking change in v2.0, your reliability library shouldn't need to care. httpx's contract is stable: send HTTP, get HTTP back.

The providers are ~80 lines each. The API surfaces for Claude and OpenAI are simple JSON-over-HTTP. The "convenience" of an SDK doesn't justify the coupling.

### Why async-first

LLM calls are I/O-bound. A typical request is 1-30 seconds of waiting for network and inference. During that time, a synchronous Python thread is doing nothing. In a web server handling multiple concurrent LLM calls, this is catastrophic for throughput.

The entire library is `async/await` native. `asyncio.wait_for` handles timeouts at the event loop level -- cleaner and more reliable than thread-based timeout hacks. The fallback chain's provider iteration is sequential by design (you don't want to burn money calling all providers in parallel), but the I/O within each call is non-blocking.

### Why Pydantic for config

`ProviderConfig` uses Pydantic `BaseModel` instead of a plain dataclass:

```python
class ProviderConfig(BaseModel):
    name: str
    timeout_seconds: float = 30.0
    max_retries: int = 2
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    api_key: str = ""
    base_url: str = ""
    model: str = ""
```

Configuration is the boundary between your code and the outside world. User-provided config, environment variables, YAML files -- all untyped at the point of entry. Pydantic validates and coerces at construction time, so invalid config fails immediately with a clear error, not halfway through a request chain when a string shows up where a float was expected.

Internal data structures (`CompletionResult`, `CircuitBreaker`, `FallbackChainConfig`) use dataclasses -- they're constructed by our own code, so the validation overhead isn't worth it.

### Why custom retry instead of tenacity

[tenacity](https://github.com/jd/tenacity) is a solid retry library. It's also the wrong abstraction for LLM retries because:

1. **Cost-aware decisions.** The retry decision isn't just "did it fail?" -- it's "did it fail, and can we afford another attempt, and is the circuit breaker still closed, and should we fall to the next provider instead?" Tenacity's `retry_if_exception_type` and `wait_exponential` don't model this.

2. **Cross-provider fallback.** Tenacity retries the same callable. Our retry exhaustion triggers a fallback to an entirely different provider with its own circuit breaker and timeout profile. This is a two-level loop, not a flat retry.

3. **Timeout adaptation.** Each retry attempt gets a different timeout based on the adaptive strategy. Tenacity doesn't have a hook for "change the timeout on each attempt."

The retry loop in `FallbackChain.complete` is 60 lines. The clarity of an explicit `for attempt in range(1, max_retries + 1)` loop with visible decision points is worth more than the abstraction tenacity provides.

---

## Cost Model

### Scenario: 1,000 requests/day, 5% primary failure rate

Assume a three-provider chain: Claude Sonnet (primary), GPT-4o (secondary), local Llama 3 (tertiary).

**Pricing (per 1K tokens, approximate):**

| Provider | Input | Output | Avg tokens/request (in/out) | Cost/request |
|----------|-------|--------|-----------------------------|-------------|
| Claude Sonnet | $3.00 | $15.00 | 500 / 300 | $0.0060 |
| GPT-4o | $2.50 | $10.00 | 500 / 300 | $0.0043 |
| Local Llama 3 | $0.00 | $0.00 | 500 / 300 | $0.0000 |

**Naive retry (no fallback, 3 retries on primary):**

```
950 requests succeed on attempt 1:          950 x $0.0060 = $5.70
 50 requests fail, retry 3x each:           150 x $0.0060 = $0.90  (wasted)
 50 requests x 33% eventual success:         17 x $0.0060 = $0.10
 33 requests remain failed:                  total failures = 33

Daily cost:  $6.70
Failed requests: 33 (3.3% effective failure rate)
Wasted spend: $0.90 on retries that mostly still fail
```

**ai-fallback chain (1 retry per provider, then fall to next):**

```
950 requests succeed on Claude attempt 1:   950 x $0.0060 = $5.70
 50 fail Claude (1 retry each):              50 x $0.0060 = $0.30  (retry cost)
 25 of those succeed on Claude retry:         25 x $0.0060 = $0.15
 25 fall to GPT-4o, 23 succeed:              23 x $0.0043 = $0.10
  2 fall to local, 2 succeed:                 2 x $0.0000 = $0.00

Daily cost:  $6.25
Failed requests: 0
Wasted spend: $0.30 (failed Claude retries only)
```

**Monthly difference:**

| Metric | Naive Retry | ai-fallback | Delta |
|--------|-------------|-------------|-------|
| Monthly cost | $201.00 | $187.50 | -$13.50/mo |
| Monthly failures | ~990 | ~0 | -990 failures |
| Wasted spend | $27.00/mo | $9.00/mo | -$18.00/mo |
| Effective uptime | 96.7% | ~99.9% | +3.2% |

The cost savings are modest at this scale. The real win is the failure rate: from 33/day to effectively zero. At enterprise scale (100K requests/day), the cost difference is $1,350/month and the failure reduction prevents thousands of degraded user experiences.

The circuit breaker adds further savings during sustained outages. If Claude is down for an hour:

- **Without circuit breaker:** 42 requests/hour x 2 retries = 84 wasted Claude API calls, each waiting for timeout.
- **With circuit breaker:** 5 failures to open the circuit, then instant fallback to GPT-4o for the remaining 37 requests. 79 fewer wasted calls. ~79 x 30s = ~40 minutes of aggregate user wait time eliminated.

---

## Failure Scenarios

| Failure Mode | Detection | Response | User Impact |
|---|---|---|---|
| Provider returns HTTP 500 | `ProviderError` with `status_code` | Retry, then fallback to next provider. Circuit breaker increments failure count. | Delayed by retry timeout, then served by fallback provider. |
| Provider returns HTTP 429 (rate limit) | `ProviderError` with `status_code=429` | Same as 500. Repeated 429s will trip circuit breaker, preventing further rate limit hits. | Automatic fallback. No manual intervention needed. |
| Provider latency spike (30s+) | `asyncio.TimeoutError` via adaptive timeout | Timeout fires at p95-adapted threshold. Retry with increased timeout, then fallback. | Timeout at adaptive threshold rather than waiting full static timeout. |
| Provider full outage (all requests fail) | Circuit breaker opens after `failure_threshold` (5) consecutive failures | Circuit opens. All subsequent requests skip to next provider instantly -- no network call, no timeout wait. Recovers via half-open after `recovery_timeout`. | First 5 requests degraded, all subsequent requests served by fallback with no latency penalty. |
| Cost runaway (retry storm) | Cost-rate circuit breaker: `$/min > cost_per_minute_limit` | Circuit opens on cost, not errors. Fallback to cheaper or free provider. | Transparent fallback. Budget protected. |
| Budget exhausted | `_check_budget()` at chain entry | `BudgetExceededError` raised before any provider call. | Hard stop. Application must handle -- queue the request, return cached response, or inform user. |
| Low confidence response | `ConfidenceGate.check()` | Escalation callback fires (e.g., Slack alert). Safe fallback response returned. Original response preserved in metadata for human review. | Receives safe default. Human reviews async. |
| Streaming truncation (connection drop) | `PartialRecovery.recover()` on available text | Trim to last complete sentence/paragraph. Return with `truncated=True` flag. | Receives partial but coherent response. Application can decide to re-request or accept. |
| All providers down | `AIFallbackError` after exhausting chain | Exception raised with all collected errors. | Application-level handling required. This is the "everything is on fire" scenario. |
| Network partition (DNS, TLS failure) | `httpx` connection error caught as `Exception` | Same path as provider error. Retry, then fallback. | Handled identically to provider errors. |

---

## File Reference

```
ai-fallback/
  src/ai_fallback/
    __init__.py
    fallback_chain.py      # Orchestrator: budget, breaker, timeout, fallback loop
    circuit_breaker.py      # Error-rate + cost-rate state machine
    timeout_strategies.py   # p95-adaptive timeout with prompt-length + backoff
    confidence_gate.py      # Threshold check, escalation callback, safe fallback
    partial_recovery.py     # Sentence/paragraph boundary trimming
    observability.py        # Structured JSONL event logging
    exceptions.py           # AIFallbackError hierarchy (6 exception types)
    providers/
      __init__.py
      base.py               # BaseProvider ABC, CompletionResult, ProviderConfig
      claude.py             # Anthropic Messages API via httpx
      openai.py             # OpenAI Chat Completions API via httpx
      local.py              # Ollama /api/generate via httpx
  tests/
  examples/
  pyproject.toml            # Python 3.10+, deps: httpx, pydantic
```

---

*This library is designed for teams that have been burned by LLM provider reliability in production and decided to stop hoping it gets better. The patterns here are not theoretical -- they're extracted from real failure modes encountered running LLM-dependent systems at scale.*
