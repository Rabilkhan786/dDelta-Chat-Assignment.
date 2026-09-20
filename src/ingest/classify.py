"""Small deterministic classifiers shared by the PDF adapters.

The project only classifies evidence it can read as text. It does not claim to
detect CAD geometry or table cells.
"""

from __future__ import annotations

import re

from src.canonical.model import ElementType

DIMENSION_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m|in|inch|bar|psi|kpa|mpa|°c|deg)\b",
    re.IGNORECASE,
)
TAG_PATTERN = re.compile(r"\b[A-Z]{1,6}[- ]?\d{1,5}[A-Z]?\b")


def classify_text(text: str, font_size: float | None = None) -> ElementType:
    """Classify only simple, inspectable text patterns."""
    compact = " ".join(text.split())
    upper = compact.upper()
    if DIMENSION_PATTERN.search(compact):
        return ElementType.DIMENSION
    if upper.startswith(("NOTE", "NTS", "GENERAL NOTE", "INSTALL", "WARNING")):
        return ElementType.NOTE
    if TAG_PATTERN.fullmatch(upper):
        return ElementType.SYMBOL
    if font_size and font_size >= 14 and len(compact) < 100:
        return ElementType.TITLE
    return ElementType.TEXT
