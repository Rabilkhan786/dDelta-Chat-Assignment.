"""Run the label-driven scorecard without fabricating ground truth."""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import precision_recall_f1
from src.observability.logging import get_logger

logger = get_logger(__name__)
DATASET = Path(__file__).with_name("datasets") / "ground_truth.json"


def main() -> None:
    """Report delta metrics for reviewer-supplied ground-truth change identifiers."""
    if not DATASET.exists():
        logger.warning("evaluation_not_run", extra={"reason": "ground_truth.json is absent; use ground_truth.template.json"})
        return
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    expected = set(dataset.get("expected_change_ids", []))
    predicted = set(dataset.get("predicted_change_ids", []))
    metrics = precision_recall_f1(predicted, expected)
    logger.info("evaluation_scorecard", extra={"precision": metrics.precision, "recall": metrics.recall,
        "f1": metrics.f1, "failure_cases": dataset.get("known_failure_cases", [])})


if __name__ == "__main__":
    main()
