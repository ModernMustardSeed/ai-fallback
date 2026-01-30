"""Partial recovery for truncated streaming responses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .providers.base import CompletionResult


@dataclass
class PartialRecovery:
    """Attempts to salvage usable content from truncated LLM responses.

    Strategies:
      - Trim to last complete sentence.
      - Trim to last complete paragraph.
      - Accept partial if it meets minimum length.
    """

    min_usable_length: int = 50
    trim_to_sentence: bool = True
    trim_to_paragraph: bool = False

    def recover(self, partial_text: str, provider_name: str) -> CompletionResult | None:
        """Try to salvage a usable result from partial text.

        Returns a CompletionResult if recovery succeeded, None if the text
        is too short or unsalvageable.
        """
        if not partial_text or len(partial_text.strip()) < self.min_usable_length:
            return None

        text = partial_text.strip()

        if self.trim_to_paragraph:
            text = self._trim_to_last_paragraph(text)
        elif self.trim_to_sentence:
            text = self._trim_to_last_sentence(text)

        if len(text) < self.min_usable_length:
            return None

        return CompletionResult(
            text=text,
            provider_used=provider_name,
            truncated=True,
            metadata={"recovered": True, "original_length": len(partial_text)},
        )

    def _trim_to_last_sentence(self, text: str) -> str:
        # Find last sentence-ending punctuation
        match = list(re.finditer(r'[.!?]\s', text))
        if match:
            return text[: match[-1].end()].strip()
        # Check if text ends with sentence-ending punctuation
        if text and text[-1] in ".!?":
            return text
        return text

    def _trim_to_last_paragraph(self, text: str) -> str:
        parts = text.rsplit("\n\n", 1)
        if len(parts) == 2 and len(parts[0].strip()) >= self.min_usable_length:
            return parts[0].strip()
        return text
