"""Tests for the separate delta, retrieval, and generation evaluation flows."""

import json

import pytest

from eval import run_eval
from src.chat.answer import GroundedAnswer
from src.chat.index import Excerpt


def test_predicted_change_ids_reads_the_generated_report(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "delta.json"
    report_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "change_type": "added",
                        "element_type": "note",
                        "page_number": 1,
                        "confidence": 1,
                        "description": "Added note",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_eval, "project_path", lambda path: report_path)

    assert len(run_eval.predicted_change_ids()) == 1


def test_predicted_change_ids_requires_a_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "project_path", lambda path: tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Delta report is missing"):
        run_eval.predicted_change_ids()


def test_delta_scorecard_compares_reviewed_ids(monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "predicted_change_ids", lambda: ["one", "extra"])
    scorecard = run_eval.evaluate_delta({"expected_change_ids": ["one"]})
    assert scorecard["precision"] == 0.5
    assert scorecard["recall"] == 1


def test_retrieval_scorecard_is_independent_from_generation(monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval,
        "search",
        lambda question: [Excerpt("evidence", "pid_a", "A", 1, "expected")],
    )
    dataset = {
        "qa_cases": [
            {
                "question": "question",
                "expected_retrieval_ids": ["expected"],
            }
        ]
    }

    scorecard = run_eval.evaluate_retrieval(dataset)

    assert scorecard["mean_recall_at_k"] == 1
    assert scorecard["mean_mrr"] == 1


def test_generation_scorecard_uses_answer_and_citation_labels(monkeypatch) -> None:
    class FakeService:
        def answer(self, question):
            return GroundedAnswer(
                "NOTE 24 was added",
                ["[delta_report | PID B | page 1 | delta-3]"],
                "request",
            )

    monkeypatch.setattr(run_eval, "GroundedChatService", FakeService)
    dataset = {
        "qa_cases": [
            {
                "question": "What note was added?",
                "expected_answer_keywords": ["NOTE 24"],
                "expected_citations": ["delta-3"],
            }
        ]
    }

    scorecard = run_eval.evaluate_generation(dataset)

    assert scorecard["answer_correctness"] == 1
    assert scorecard["mean_citation_accuracy"] == 1


def test_generation_provider_error_is_not_counted_as_a_score(monkeypatch) -> None:
    class FailingService:
        def answer(self, question):
            return GroundedAnswer("provider failed", [], "request", "provider_error")

    monkeypatch.setattr(run_eval, "GroundedChatService", FailingService)
    assert run_eval.evaluate_generation({"qa_cases": [{"question": "question"}]}) is None


def test_empty_evaluation_sections_are_reported_as_not_run() -> None:
    assert run_eval.evaluate_retrieval({"qa_cases": []}) is None
    assert run_eval.evaluate_generation({"qa_cases": []}) is None


def test_missing_dataset_stops_before_reading_a_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "DATASET", tmp_path / "missing.json")
    monkeypatch.setattr(
        run_eval,
        "predicted_change_ids",
        lambda: (_ for _ in ()).throw(AssertionError("report should not be read")),
    )

    run_eval.main(write_candidates=False)


def test_main_can_write_candidates_and_skip_generation(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "ground_truth.json"
    dataset_path.write_text(json.dumps({"known_failure_cases": []}), encoding="utf-8")
    monkeypatch.setattr(run_eval, "DATASET", dataset_path)
    monkeypatch.setattr(run_eval, "predicted_change_ids", lambda: ["change-b", "change-a"])
    monkeypatch.setattr(run_eval, "evaluate_delta", lambda dataset: {})
    monkeypatch.setattr(run_eval, "evaluate_retrieval", lambda dataset: {})
    monkeypatch.setattr(
        run_eval,
        "evaluate_generation",
        lambda dataset: (_ for _ in ()).throw(AssertionError("generation should be skipped")),
    )

    run_eval.main(write_candidates=True, skip_generation=True)

    candidates = json.loads((tmp_path / "predicted_change_ids.json").read_text(encoding="utf-8"))
    assert candidates["predicted_change_ids"] == ["change-a", "change-b"]
