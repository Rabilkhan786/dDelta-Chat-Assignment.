"""Run labelled delta, retrieval, and answer scorecards without making labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import (
    answer_correct,
    citation_accuracy,
    citation_coverage,
    mean_reciprocal_rank,
    precision_recall_f1,
    report_entry_id,
    retrieval_recall_at_k,
)
from src.chat.answer import GroundedChatService
from src.chat.index import search
from src.config.settings import project_path, settings
from src.observability.logging import get_logger

logger = get_logger(__name__)
DATASET = Path(__file__).with_name("datasets") / "ground_truth.json"


def predicted_change_ids() -> list[str]:
    """Read non-unchanged IDs from the generated report."""
    report_path = project_path(settings.paths.delta_json)
    if not report_path.exists():
        raise FileNotFoundError(
            "Delta report is missing. Run `python main.py run` before evaluation."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [
        report_entry_id(entry)
        for entry in report.get("entries", [])
        if entry["change_type"] != "unchanged"
    ]


def evaluate_delta(dataset: dict) -> None:
    """Score deterministic delta changes against the reviewed expected IDs."""
    predicted = set(predicted_change_ids())
    expected = set(dataset.get("expected_change_ids", []))
    metrics = precision_recall_f1(predicted, expected)
    logger.info(
        "delta_scorecard",
        extra={
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "predicted": len(predicted),
            "expected": len(expected),
        },
    )


def evaluate_retrieval(dataset: dict) -> None:
    """Score search alone so a missing LLM key cannot hide retrieval quality."""
    results = []
    for case in dataset.get("qa_cases", []):
        expected_ids = case.get("expected_retrieval_ids", [])
        if not expected_ids:
            continue
        result_ids = [item.element_id for item in search(case["question"])]
        results.append(
            {
                "question": case["question"],
                "recall_at_k": retrieval_recall_at_k(
                    result_ids, expected_ids, settings.retrieval.top_k
                ),
                "mrr": mean_reciprocal_rank(result_ids, expected_ids),
            }
        )
    if not results:
        logger.warning(
            "retrieval_evaluation_not_run",
            extra={
                "reason": "qa_cases have no reviewed expected_retrieval_ids",
            },
        )
        return
    logger.info(
        "retrieval_scorecard",
        extra={
            "cases": len(results),
            "mean_recall_at_k": sum(item["recall_at_k"] for item in results) / len(results),
            "mean_mrr": sum(item["mrr"] for item in results) / len(results),
            "results": results,
        },
    )


def evaluate_generation(dataset: dict) -> None:
    """Score LLM text and citations after retrieval; needs a configured provider."""
    cases = dataset.get("qa_cases", [])
    if not cases:
        logger.warning("generation_evaluation_not_run", extra={"reason": "qa_cases is empty"})
        return
    service = GroundedChatService()
    results = []
    for case in cases:
        answer = service.answer(case["question"])
        if answer.status == "provider_error":
            logger.warning(
                "generation_evaluation_not_run",
                extra={
                    "reason": "LLM provider is unavailable; retrieval metrics still ran",
                },
            )
            return
        results.append(
            {
                "question": case["question"],
                "correct": answer_correct(answer.text, case.get("expected_answer_keywords", [])),
                "citation_accuracy": citation_accuracy(
                    answer.citations, case.get("expected_citations", [])
                ),
                "citation_coverage": citation_coverage(
                    answer.citations, case.get("expected_citations", [])
                ),
                "status": answer.status,
            }
        )
    logger.info(
        "chat_scorecard",
        extra={
            "cases": len(results),
            "answer_correctness": sum(item["correct"] for item in results) / len(results),
            "mean_citation_accuracy": sum(item["citation_accuracy"] for item in results)
            / len(results),
            "mean_citation_coverage": sum(item["citation_coverage"] for item in results)
            / len(results),
            "results": results,
        },
    )


def main(write_candidates: bool, skip_generation: bool = False) -> None:
    """Print all independent scorecards; candidate output never edits labels."""
    predicted = set(predicted_change_ids())
    if write_candidates:
        candidate_file = DATASET.with_name("predicted_change_ids.json")
        candidate_file.write_text(
            json.dumps({"predicted_change_ids": sorted(predicted)}, indent=2), encoding="utf-8"
        )
        logger.info(
            "evaluation_candidates_written",
            extra={"path": str(candidate_file), "count": len(predicted)},
        )
    if not DATASET.exists():
        logger.warning("evaluation_not_run", extra={"reason": "ground_truth.json is absent"})
        return
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    evaluate_delta(dataset)
    logger.info("known_failure_cases", extra={"cases": dataset.get("known_failure_cases", [])})
    evaluate_retrieval(dataset)
    if skip_generation:
        logger.info("generation_evaluation_not_run", extra={"reason": "--skip-generation"})
        return
    try:
        evaluate_generation(dataset)
    except Exception:
        logger.exception(
            "generation_evaluation_failed",
            extra={"reason": "a configured LLM provider is required"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated deltas and grounded chat.")
    parser.add_argument(
        "--write-candidates", action="store_true", help="write candidate IDs for human review"
    )
    parser.add_argument(
        "--skip-generation", action="store_true", help="evaluate without an LLM call"
    )
    arguments = parser.parse_args()
    main(arguments.write_candidates, arguments.skip_generation)
