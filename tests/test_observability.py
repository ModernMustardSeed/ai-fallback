"""Tests for LLMObserver."""

import io
from ai_fallback import LLMObserver, CompletionResult


def test_log_attempt():
    obs = LLMObserver()
    obs.log_attempt("claude", "Hello world")
    assert len(obs.events) == 1
    assert obs.events[0]["event"] == "attempt"
    assert obs.events[0]["provider"] == "claude"


def test_log_success():
    obs = LLMObserver()
    result = CompletionResult(
        text="hi", provider_used="claude", model="claude-3", latency_ms=150.0,
        cost_incurred=0.001, input_tokens=10, output_tokens=5, attempts=1,
    )
    obs.log_success(result)
    assert obs.events[0]["event"] == "success"
    assert obs.events[0]["latency_ms"] == 150.0


def test_log_to_stream():
    buf = io.StringIO()
    obs = LLMObserver(stream=buf)
    obs.log_failure("openai", "timeout", attempt=2)
    output = buf.getvalue()
    assert "failure" in output
    assert "openai" in output


def test_log_fallback():
    obs = LLMObserver()
    obs.log_fallback("claude", "openai", "timeout")
    assert obs.events[0]["event"] == "fallback"


def test_log_circuit_open():
    obs = LLMObserver()
    obs.log_circuit_open("claude", "too many failures")
    assert obs.events[0]["event"] == "circuit_open"
