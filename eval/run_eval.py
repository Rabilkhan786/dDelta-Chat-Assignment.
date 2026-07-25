"""Run the label-driven scorecard without fabricating ground truth."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from eval.metrics import precision_recall_f1, report_entry_id
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


def main(write_candidates: bool = False) -> None:
    """Report delta metrics for reviewer-supplied ground-truth change identifiers."""
    predicted = set(predicted_change_ids())
    if write_candidates:
        candidate_file = DATASET.with_name("predicted_change_ids.json")
        candidate_file.write_text(json.dumps({"predicted_change_ids": sorted(predicted)}, indent=2), encoding="utf-8")
        logger.info("evaluation_candidates_written", extra={"path": str(candidate_file), "count": len(predicted)})
    if not DATASET.exists():
        logger.warning("evaluation_not_run", extra={"reason": "ground_truth.json is absent; use ground_truth.template.json"})
        return
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    expected = set(dataset.get("expected_change_ids", []))
    metrics = precision_recall_f1(predicted, expected)
    logger.info("evaluation_scorecard", extra={"precision": metrics.precision, "recall": metrics.recall,
        "f1": metrics.f1, "failure_cases": dataset.get("known_failure_cases", [])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated delta IDs against human labels.")
    parser.add_argument("--write-candidates", action="store_true", help="write generated IDs for human labelling")
    main(parser.parse_args().write_candidates)
