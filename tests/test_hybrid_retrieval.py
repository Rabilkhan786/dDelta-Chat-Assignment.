"""Unit tests for the deterministic pieces of hybrid retrieval."""

from src.chat.index import reciprocal_rank_fusion, route_question


def test_rrf_rewards_an_identifier_found_by_both_retrievers() -> None:
    scores = reciprocal_rank_fusion([["PSV-9066", "P-101"], ["PSV-9066", "note"]], rrf_k=10)
    assert scores["PSV-9066"] > scores["P-101"]


def test_change_question_prefers_delta_report() -> None:
    assert route_question("What changed on PSV-9066?") == {"delta_report"}


def test_comparison_question_keeps_all_sources() -> None:
    assert route_question("Compare revision A and revision B") is None
