"""Persistence helpers for the canonical architectural seam."""

from pathlib import Path

from src.canonical.model import CanonicalDocument


def write_canonical_document(document: CanonicalDocument, destination: Path) -> None:
    """Write a canonical document as stable, machine-readable JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def read_canonical_document(source: Path) -> CanonicalDocument:
    """Load a canonical document without exposing its original format downstream."""
    return CanonicalDocument.model_validate_json(source.read_text(encoding="utf-8"))
