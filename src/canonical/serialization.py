"""Persistence helpers for the canonical architectural seam."""

from pathlib import Path

from src.canonical.model import CanonicalDocument


def write_canonical_document(document: CanonicalDocument, destination: Path) -> None:
    """Write a canonical document as stable, machine-readable JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(document.model_dump_json(indent=2))
