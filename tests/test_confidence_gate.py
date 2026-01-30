"""Tests for ConfidenceGate."""

import pytest
from ai_fallback import ConfidenceGate, CompletionResult, LowConfidenceError


def _result(confidence):
    return CompletionResult(text="answer", provider_used="test", confidence=confidence)


@pytest.mark.asyncio
async def test_passes_above_threshold():
    gate = ConfidenceGate(min_threshold=0.5)
    result = await gate.check(_result(0.8))
    assert result.text == "answer"


@pytest.mark.asyncio
async def test_passes_none_confidence():
    gate = ConfidenceGate(min_threshold=0.5)
    result = await gate.check(_result(None))
    assert result.text == "answer"


@pytest.mark.asyncio
async def test_raises_below_threshold():
    gate = ConfidenceGate(min_threshold=0.7)
    with pytest.raises(LowConfidenceError):
        await gate.check(_result(0.3))


@pytest.mark.asyncio
async def test_fallback_response():
    gate = ConfidenceGate(min_threshold=0.7, fallback_response="I'm not sure.")
    result = await gate.check(_result(0.3))
    assert result.text == "I'm not sure."
    assert result.provider_used == "confidence_gate_fallback"


@pytest.mark.asyncio
async def test_escalation_callback():
    escalated = []

    async def on_escalate(r):
        escalated.append(r)

    gate = ConfidenceGate(
        min_threshold=0.7,
        escalation_callback=on_escalate,
        fallback_response="fallback",
    )
    await gate.check(_result(0.2))
    assert len(escalated) == 1
    assert escalated[0].confidence == 0.2
