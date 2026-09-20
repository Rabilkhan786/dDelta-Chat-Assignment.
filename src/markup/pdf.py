"""Write a small visual overlay for the reliable text-bounding-box deltas."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.canonical.model import DeltaEntry, DeltaType

COLORS = {
    DeltaType.ADDED: (0, 0.6, 0),
    DeltaType.REMOVED: (0.9, 0, 0),
    DeltaType.MODIFIED: (1, 0.55, 0),
    DeltaType.MOVED: (0.3, 0.2, 0.9),
}


def write_markup(source_pdf: Path, destination: Path, deltas: list[DeltaEntry]) -> Path:
    """Box changed text regions on Revision B; skipped entries have no reliable box."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source_pdf) as document:
        for delta in deltas:
            if delta.change_type == DeltaType.UNCHANGED or not delta.region:
                continue
            page = document[delta.page_number - 1]
            box = fitz.Rect(delta.region.x0, delta.region.y0, delta.region.x1, delta.region.y1)
            annotation = page.add_rect_annot(box)
            annotation.set_colors(stroke=COLORS[delta.change_type])
            annotation.set_info(title=f"Delta: {delta.change_type.value}", content=delta.description)
            annotation.update()
        document.save(destination)
    return destination
