"""Run the label-driven scorecard without fabricating ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import answer_correct, citation_accuracy, precision_recall_f1, report_entry_id
from src.chat.answer import GroundedChatService
from src.config.settings import project_path, settings
from src.observability.logging import get_logger

logger = get_logger(__name__)
DATASET = Path(__file__).with_name("datasets") / "ground_truth.json"


def predicted_change_ids() -> list[str]:
    """Derive evaluator IDs directly from the generated deterministic report."""
    report_path = project_path(settings.paths.delta_json)
    if not report_path.exists():
        raise FileNotFoundError("Delta report is missing. Run `python main.py run` before evaluation.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [report_entry_id(entry) for entry in report.get("entries", []) if entry["change_type"] != "unchanged"]


def evaluate_delta(dataset: dict) -> None:
    """Precision/recall/F1 of detected change IDs against human-labeled ones."""
    predicted = set(predicted_change_ids())
    expected = set(dataset.get("expected_change_ids", []))
    metrics = precision_recall_f1(predicted, expected)
    logger.info("delta_scorecard", extra={"precision": metrics.precision, "recall": metrics.recall,
        "f1": metrics.f1, "predicted": len(predicted), "expected": len(expected)})


def evaluate_chat(dataset: dict) -> None:
    """Answer correctness and citation accuracy over human-labeled questions."""
    qa_cases = dataset.get("qa_cases", [])
    if not qa_cases:
        logger.warning("chat_evaluation_not_run", extra={"reason": "qa_cases is empty in ground_truth.json"})
        return

    service = GroundedChatService()
    results = []
    for case in qa_cases:
        answer = service.answer(case["question"])
        correct = answer_correct(answer.text, case.get("expected_answer_keywords", []))
        accuracy = citation_accuracy(answer.citations, case.get("expected_citations", []))
        results.append({"question": case["question"], "correct": correct, "citation_accuracy": accuracy})

    logger.info("chat_scorecard", extra={
        "cases": len(results),
        "answer_correctness": sum(r["correct"] for r in results) / len(results),
        "mean_citation_accuracy": sum(r["citation_accuracy"] for r in results) / len(results),
        "results": results,
    })


def main(write_candidates: bool) -> None:
    predicted = set(predicted_change_ids())
    if write_candidates:
        candidate_file = DATASET.with_name("predicted_change_ids.json")
        candidate_file.write_text(json.dumps({"predicted_change_ids": sorted(predicted)}, indent=2), encoding="utf-8")
        logger.info("evaluation_candidates_written", extra={"path": str(candidate_file), "count": len(predicted)})

    if not DATASET.exists():
        logger.warning("evaluation_not_run", extra={"reason": "ground_truth.json is absent; use ground_truth.template.json"})
        return

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    evaluate_delta(dataset)
    logger.info("known_failure_cases", extra={"cases": dataset.get("known_failure_cases", [])})
    try:
        evaluate_chat(dataset)
    except Exception:
        logger.exception("chat_evaluation_failed", extra={"reason": "grounded chat requires a configured LLM provider"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated deltas and grounded chat against human labels.")
    parser.add_argument("--write-candidates", action="store_true", help="write generated delta IDs for human labelling")
    main(parser.parse_args().write_candidates)
