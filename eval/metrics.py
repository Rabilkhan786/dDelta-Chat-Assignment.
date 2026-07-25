"""Deterministic metrics used by the assignment evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass


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
