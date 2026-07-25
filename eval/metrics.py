"""Deterministic metrics used by the assignment evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float


def precision_recall_f1(predicted: set[str], expected: set[str]) -> ClassificationMetrics:
    """Measure exact change IDs; labels must be supplied by a human reviewer."""
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(precision, recall, f1)


def citation_accuracy(citations: list[str], expected: set[str]) -> float:
    """Measure whether the answer cites at least one expected evidence locator."""
    return len(set(citations) & expected) / len(expected) if expected else 0.0


def report_entry_id(entry: dict[str, Any]) -> str:
    """Create a stable evaluator ID from the machine-readable delta entry."""
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
