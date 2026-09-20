"""Tests for deterministic technical-identifier tokenization."""

from src.chat.query import keyword_tokens


def test_keyword_tokens_match_common_identifier_formats() -> None:
    compact = set(keyword_tokens("PSV9066A"))
    spaced = set(keyword_tokens("PSV 9066A"))
    hyphenated = set(keyword_tokens("PSV-9066A"))

    assert {"psv", "9066a"} <= compact & spaced & hyphenated
    assert "1.5" in keyword_tokens("1.5 bar")
