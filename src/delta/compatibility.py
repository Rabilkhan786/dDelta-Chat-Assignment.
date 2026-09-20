"""A small warning signal for pairs that may not be document revisions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.canonical.model import CanonicalDocument
from src.config.settings import settings

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-/]\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class CompatibilityResult:
    """Revision similarity, intended as a warning rather than a hidden decision."""

    score: float
    compatible: bool
    message: str | None


def check_revision_compatibility(old_document: CanonicalDocument, new_document: CanonicalDocument,
                                 minimum_similarity: float | None = None) -> CompatibilityResult:
    """Compare unique text tokens; low overlap warns before a delta is trusted."""
    old_tokens = _tokens(old_document)
    new_tokens = _tokens(new_document)
    union = old_tokens | new_tokens
    score = len(old_tokens & new_tokens) / len(union) if union else 0.0
    threshold = minimum_similarity if minimum_similarity is not None else settings.compatibility.minimum_similarity
    compatible = score >= threshold
    message = None if compatible else (
        "Low document similarity: these inputs may be different systems rather than revisions. "
        "Review this delta before using it."
    )
    return CompatibilityResult(round(score, 3), compatible, message)


def _tokens(document: CanonicalDocument) -> set[str]:
    return {token.lower() for page in document.pages for element in page.elements
            for token in TOKEN_PATTERN.findall(element.text) if len(token) > 1}
