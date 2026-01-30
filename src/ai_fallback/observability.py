"""Structured observability for LLM calls."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import IO

from .providers.base import CompletionResult

logger = logging.getLogger("ai_fallback")


class LLMObserver:
    """Structured JSONL logger for LLM call observability."""

    def __init__(
        self,
        log_file: str | Path | None = None,
        stream: IO | None = None,
    ):
        self._file: IO | None = None
        if log_file:
            self._file = open(log_file, "a", encoding="utf-8")
        self._stream = stream
        self._events: list[dict] = []

    def log_attempt(
        self,
        provider_name: str,
        prompt_preview: str,
        *,
        attempt: int = 1,
    ) -> None:
        self._emit(
            event="attempt",
            provider=provider_name,
            attempt=attempt,
            prompt_preview=prompt_preview[:120],
        )

    def log_success(self, result: CompletionResult) -> None:
        self._emit(
            event="success",
            provider=result.provider_used,
            model=result.model,
            latency_ms=round(result.latency_ms, 1),
            cost=round(result.cost_incurred, 6),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            attempts=result.attempts,
            truncated=result.truncated,
        )

    def log_failure(
        self,
        provider_name: str,
        error: str,
        *,
        attempt: int = 1,
    ) -> None:
        self._emit(
            event="failure",
            provider=provider_name,
            error=error,
            attempt=attempt,
        )

    def log_circuit_open(self, provider_name: str, reason: str) -> None:
        self._emit(event="circuit_open", provider=provider_name, reason=reason)

    def log_fallback(self, from_provider: str, to_provider: str, reason: str) -> None:
        self._emit(
            event="fallback",
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
        )

    def log_budget_warning(self, current_spend: float, limit: float) -> None:
        self._emit(
            event="budget_warning",
            current_spend=round(current_spend, 6),
            limit=round(limit, 6),
        )

    def _emit(self, **data: object) -> None:
        record = {"ts": time.time(), **data}
        self._events.append(record)
        line = json.dumps(record)
        logger.debug(line)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        if self._stream:
            self._stream.write(line + "\n")
            self._stream.flush()

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
