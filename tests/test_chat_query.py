"""Tests for deterministic query tokenization and source routing."""

import pytest

from src.chat.query import is_broad_change_question, keyword_tokens, route_question


def test_keyword_tokens_match_common_identifier_formats() -> None:
    compact = set(keyword_tokens("PSV9066A"))
    spaced = set(keyword_tokens("PSV 9066A"))
    hyphenated = set(keyword_tokens("PSV-9066A"))

    assert {"psv", "9066a"} <= compact & spaced & hyphenated
    assert "1.5" in keyword_tokens("1.5 bar")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What changed in revision B?", "delta_report"),
        ("What does revision A say?", "pid_a"),
        ("Show revision B", "pid_b"),
        ("Compare revision A and revision B", None),
        ("Show the heat exchanger", None),
    ],
)
def test_route_question_uses_explicit_intent_words(query: str, expected: str | None) -> None:
    assert route_question(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What changed?", True),
        ("Summarize all changes in revision B", True),
        ("What changed on PSV-9066?", False),
        ("What callout was removed?", False),
        ("Compare revision A and revision B", False),
    ],
)
def test_broad_change_question_detection(query: str, expected: bool) -> None:
    assert is_broad_change_question(query) is expected
