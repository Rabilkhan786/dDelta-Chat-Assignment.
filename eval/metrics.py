"""Deterministic metrics used by the assignment evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
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


def report_entry_id(entry: dict[str, Any]) -> str:
    """Create a stable ID without invalidating reviewed labels when report metadata grows."""
    core_keys = ("change_type", "element_type", "page_number", "confidence", "description", "bounding_box")
    payload = json.dumps({key: entry.get(key) for key in core_keys if key in entry},
        sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def answer_correct(answer_text: str, expected_keywords: list[str]) -> bool:
    """A human-labeled answer is 'correct' if every expected keyword appears in it."""
    if not expected_keywords:
        return False
    lowered = answer_text.lower()
    return all(keyword.lower() in lowered for keyword in expected_keywords)


def citation_accuracy(citations: list[str], expected_fragments: list[str]) -> float:
    """Fraction of expected citation fragments (e.g. "pid_a | page 3") found in the answer's citations."""
    if not expected_fragments:
        return 0.0
    hits = sum(1 for fragment in expected_fragments if any(fragment in citation for citation in citations))
    return hits / len(expected_fragments)


def retrieval_recall_at_k(result_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Measure whether human-labelled evidence appears in the first k results."""
    expected = set(expected_ids)
    if not expected:
        return 0.0
    return len(set(result_ids[:k]) & expected) / len(expected)


def mean_reciprocal_rank(result_ids: list[str], expected_ids: list[str]) -> float:
    """Score the rank of the first human-labelled relevant excerpt."""
    expected = set(expected_ids)
    for position, result_id in enumerate(result_ids, start=1):
        if result_id in expected:
            return 1 / position
    return 0.0
