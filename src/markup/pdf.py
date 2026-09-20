"""Write a small visual overlay for the reliable text-bounding-box deltas."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.canonical.model import DeltaEntry, DeltaType
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)

COLORS = {
    DeltaType.ADDED: (0, 0.6, 0),
    DeltaType.REMOVED: (0.9, 0, 0),
    DeltaType.MODIFIED: (1, 0.55, 0),
    DeltaType.MOVED: (0.3, 0.2, 0.9),
}


def write_markup(source_pdf: Path, destination: Path, deltas: list[DeltaEntry]) -> Path:
    """Box changed text regions on Revision B; skipped entries have no reliable box."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with stage(logger, "delta_markup"), fitz.open(source_pdf) as document:
        for delta in deltas:
            if delta.change_type == DeltaType.UNCHANGED or not delta.region:
                continue
            if not 1 <= delta.page_number <= len(document):
                logger.warning(
                    "markup_region_skipped",
                    extra={"page": delta.page_number, "reason": "page_not_in_revision_b"},
                )
                continue
            page = document[delta.page_number - 1]
            box = fitz.Rect(delta.region.x0, delta.region.y0, delta.region.x1, delta.region.y1)
            annotation = page.add_rect_annot(box)
            annotation.set_colors(stroke=COLORS[delta.change_type])
            annotation.set_info(
                title=f"Delta: {delta.change_type.value}", content=delta.description
            )
            annotation.update()
        document.save(destination)
    return destination
