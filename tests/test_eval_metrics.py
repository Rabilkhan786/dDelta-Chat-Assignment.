"""Hand-calculated metric examples, independent of the sample ground-truth file."""

import pytest

from eval.metrics import (
    answer_correct,
    citation_accuracy,
    citation_coverage,
    mean_reciprocal_rank,
    precision_recall_f1,
    retrieval_recall_at_k,
)


def test_delta_scores_penalize_false_positives_and_missed_changes():
    result = precision_recall_f1({"correct", "extra"}, {"correct", "missed", "also-missed"})
    assert result.precision == 0.5
    assert result.recall == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(0.4)


def test_retrieval_scores_respect_rank_limit_and_duplicates():
    results = ["wrong", "a", "a", "b"]
    assert retrieval_recall_at_k(results, ["a", "b"], 3) == 0.5
    assert mean_reciprocal_rank(results, ["a", "b"]) == 0.5
    assert mean_reciprocal_rank(results, ["missing"]) == 0


def test_citation_precision_penalizes_unexpected_sources():
    citations = ["[pid_a | page 1]", "[pid_b | page 2]"]
    assert citation_accuracy(citations, ["pid_a"]) == 0.5
    assert citation_coverage(citations, ["pid_a"]) == 1
    assert citation_coverage(citations, ["pid_a", "delta_report"]) == 0.5


def test_missing_labels_and_empty_answers_are_not_scored_as_success():
    assert not answer_correct("some text", [])
    assert not answer_correct("", ["pressure"])
    assert citation_accuracy([], ["pid_a"]) == 0
    assert citation_coverage([], []) == 0
    assert retrieval_recall_at_k([], [], 5) == 0
