"""Tests for PartialRecovery."""

from ai_fallback import PartialRecovery


def test_recovers_complete_sentence():
    pr = PartialRecovery(min_usable_length=10)
    result = pr.recover("This is a complete sentence. And this is trunc", "test")
    assert result is not None
    assert result.text == "This is a complete sentence."
    assert result.truncated is True


def test_returns_none_for_short_text():
    pr = PartialRecovery(min_usable_length=50)
    result = pr.recover("Short.", "test")
    assert result is None


def test_returns_none_for_empty():
    pr = PartialRecovery()
    assert pr.recover("", "test") is None
    assert pr.recover(None, "test") is None  # type: ignore


def test_paragraph_trim():
    pr = PartialRecovery(min_usable_length=10, trim_to_paragraph=True)
    text = "First paragraph is complete and long enough.\n\nSecond paragraph is trunc"
    result = pr.recover(text, "test")
    assert result is not None
    assert "Second" not in result.text


def test_no_trim():
    pr = PartialRecovery(min_usable_length=5, trim_to_sentence=False)
    result = pr.recover("Just some text without period", "test")
    assert result is not None
    assert result.text == "Just some text without period"
