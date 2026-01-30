"""Local/Ollama provider via httpx."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from ..exceptions import ProviderError
from .base import BaseProvider, CompletionResult, ProviderConfig


class LocalProvider(BaseProvider):
    """Ollama-compatible local model provider."""

    def __init__(self, config: ProviderConfig | None = None, **kwargs):
        if config is None:
            config = ProviderConfig(
                name="local",
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                model=kwargs.get("model", "llama3"),
                cost_per_input_token=0.0,
                cost_per_output_token=0.0,
                **{k: v for k, v in kwargs.items() if k in ProviderConfig.model_fields and k not in ("base_url", "model", "cost_per_input_token", "cost_per_output_token")},
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
        payload: dict = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient() as client:
            t0 = time.monotonic()
            resp = await client.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        elapsed = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            raise ProviderError(self.name, resp.text, status_code=resp.status_code)

        data = resp.json()
        return CompletionResult(
            text=data.get("response", ""),
            provider_used=self.name,
            model=data.get("model", self.config.model),
            latency_ms=elapsed,
            cost_incurred=0.0,
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise ProviderError(self.name, body.decode(), status_code=resp.status_code)
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
