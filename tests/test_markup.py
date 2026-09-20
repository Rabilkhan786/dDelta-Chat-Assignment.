"""Verify the optional PDF overlay writes visible annotations."""

import fitz

from src.canonical.model import BoundingBox, DeltaEntry, DeltaType, ElementType
from src.markup.pdf import write_markup


def test_markup_adds_one_annotation_for_a_changed_bbox(tmp_path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "markup.pdf"
    with fitz.open() as document:
        document.new_page(width=200, height=200)
        document.save(source)
    delta = DeltaEntry(change_type=DeltaType.ADDED, element_type=ElementType.TEXT, page_number=1,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25), description="Added text: 'new note'", confidence=1)
    write_markup(source, destination, [delta])
    with fitz.open(destination) as marked:
        assert marked[0].first_annot is not None
