"""Regression cases for conservative rewriting and source selection."""

import pytest

from src.chat.query import keyword_tokens, normalize_query, route_question, search_queries


@pytest.mark.parametrize(
    "original, expected",
    [
        ("PSV 009066A", "PSV-009066A"),
        ("PSV9066A", "PSV-9066A"),
        ("P–101 and DN 150", "P-101 and DN150"),
        (
            "Do not change PSV-9066B from 1.5 bar in revision A",
            "Do not change PSV-9066B from 1.5 bar in revision A",
        ),
        ("pressure safety valve 9066", "pressure safety valve 9066"),
        ("Was it moved?", "Was it moved?"),
    ],
)
def test_rewriting_preserves_meaning_and_numbers(original, expected):
    assert normalize_query(original) == expected


def test_original_query_is_always_searched():
    question = "What changed on PSV 009066A?"
    assert search_queries(question) == [question, "What changed on PSV-009066A?"]
    assert search_queries(question, rewrite=False) == [question]
    assert search_queries("PSV-9066A") == ["PSV-9066A"]


def test_keyword_variants_share_tokens_but_different_suffixes_do_not():
    assert keyword_tokens("PSV9066A") == keyword_tokens("PSV 9066A")
    assert "psv-9066a" in keyword_tokens("PSV-9066A")
    assert "9066a" in keyword_tokens("PSV-9066A")
    assert "9066b" not in keyword_tokens("PSV-9066A")
    assert "1.5" in keyword_tokens("1.5 bar")


@pytest.mark.parametrize(
    "question, source",
    [
        ("Show the heat exchanger", None),
        ("What was unchanged?", None),
        ("Tell me about PID A and PID B", None),
        ("Difference between both revisions", None),
        ("What changed on P-101?", {"delta_report"}),
        ("Pressure in rev. A?", {"pid_a"}),
        ("Pressure in revision B?", {"pid_b"}),
    ],
)
def test_routing_matches_words_and_handles_both_revisions(question, source):
    assert route_question(question) == source
