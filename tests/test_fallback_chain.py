"""Tests for FallbackChain."""

import pytest
from ai_fallback import FallbackChain, FallbackChainConfig, BudgetExceededError, AIFallbackError
from tests.conftest import MockProvider


@pytest.mark.asyncio
async def test_first_provider_succeeds(mock_ok):
    chain = FallbackChain([mock_ok])
    result = await chain.complete("Hello")
    assert result.text == "Hello from mock"
    assert result.provider_used == "ok"


@pytest.mark.asyncio
async def test_falls_back_on_failure(mock_fail, mock_ok):
    chain = FallbackChain([mock_fail, mock_ok])
    result = await chain.complete("Hello")
    assert result.provider_used == "ok"
    assert mock_fail.call_count == 2  # max_retries=2


@pytest.mark.asyncio
async def test_all_providers_fail(mock_fail):
    fail2 = MockProvider(name="fail2", fail=True)
    chain = FallbackChain([mock_fail, fail2])
    with pytest.raises(AIFallbackError, match="All 2 providers failed"):
        await chain.complete("Hello")


@pytest.mark.asyncio
async def test_budget_enforcement():
    expensive = MockProvider(name="exp", cost=0.5)
    chain = FallbackChain(
        [expensive],
        config=FallbackChainConfig(budget_limit=0.4),
    )
    # First call succeeds (0.5 spent)
    await chain.complete("Hello")
    # Second call should fail budget check
    with pytest.raises(BudgetExceededError):
        await chain.complete("Hello again")


@pytest.mark.asyncio
async def test_cost_tracking():
    p = MockProvider(name="p", cost=0.01)
    chain = FallbackChain([p])
    await chain.complete("a")
    await chain.complete("b")
    assert chain.total_cost == pytest.approx(0.02)
