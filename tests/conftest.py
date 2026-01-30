"""Shared test fixtures — mock providers."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from ai_fallback.providers.base import BaseProvider, CompletionResult, ProviderConfig


class MockProvider(BaseProvider):
    """Configurable mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        response: str = "mock response",
        cost: float = 0.001,
        confidence: float | None = None,
        fail: bool = False,
        fail_message: str = "mock failure",
    ):
        super().__init__(ProviderConfig(name=name, max_retries=2))
        self._response = response
        self._cost = cost
        self._confidence = confidence
        self._fail = fail
        self._fail_message = fail_message
        self.call_count = 0

    async def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.7):
        self.call_count += 1
        if self._fail:
            raise RuntimeError(self._fail_message)
        return CompletionResult(
            text=self._response,
            provider_used=self.name,
            model="mock-model",
            cost_incurred=self._cost,
            input_tokens=10,
            output_tokens=20,
            confidence=self._confidence,
        )

    async def stream(self, prompt, *, system=None, max_tokens=1024, temperature=0.7):
        if self._fail:
            raise RuntimeError(self._fail_message)
        for word in self._response.split():
            yield word + " "


@pytest.fixture
def mock_ok():
    return MockProvider(name="ok", response="Hello from mock")


@pytest.fixture
def mock_fail():
    return MockProvider(name="fail", fail=True)


@pytest.fixture
def mock_expensive():
    return MockProvider(name="expensive", cost=1.0, response="Expensive answer")


@pytest.fixture
def mock_low_confidence():
    return MockProvider(name="low_conf", confidence=0.3, response="Unsure answer")
