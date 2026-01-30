"""OpenAI provider via raw httpx."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from ..exceptions import ProviderError
from .base import BaseProvider, CompletionResult, ProviderConfig


class OpenAIProvider(BaseProvider):
    """OpenAI API provider using httpx (no SDK dependency)."""

    def __init__(self, config: ProviderConfig | None = None, **kwargs):
        if config is None:
            config = ProviderConfig(
                name="openai",
                base_url="https://api.openai.com/v1",
                model=kwargs.get("model", "gpt-4o"),
                api_key=kwargs.get("api_key", ""),
                cost_per_input_token=kwargs.get("cost_per_input_token", 2.5e-6),
                cost_per_output_token=kwargs.get("cost_per_output_token", 10e-6),
                **{k: v for k, v in kwargs.items() if k in ProviderConfig.model_fields and k not in ("model", "api_key", "cost_per_input_token", "cost_per_output_token")},
            )
        super().__init__(config)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> CompletionResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            t0 = time.monotonic()
            resp = await client.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        elapsed = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            raise ProviderError(self.name, resp.text, status_code=resp.status_code)

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return CompletionResult(
            text=text,
            provider_used=self.name,
            model=data.get("model", self.config.model),
            latency_ms=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_incurred=self.estimate_cost(input_tokens, output_tokens),
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise ProviderError(self.name, body.decode(), status_code=resp.status_code)
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        event = json.loads(line[6:])
                        delta = event["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
