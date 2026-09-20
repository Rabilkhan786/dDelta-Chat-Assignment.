"""Verify the optional PDF overlay writes visible annotations."""

import pymupdf

from src.canonical.model import BoundingBox, DeltaEntry, DeltaType, ElementType
from src.markup.pdf import write_markup


def test_markup_adds_one_annotation_for_a_changed_bbox(tmp_path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "markup.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=200)
        document.save(source)
    delta = DeltaEntry(
        change_type=DeltaType.ADDED,
        element_type=ElementType.TEXT,
        page_number=1,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Added text: 'new note'",
        confidence=1,
    )
    write_markup(source, destination, [delta])
    with pymupdf.open(destination) as marked:
        assert marked[0].first_annot is not None


def test_markup_skips_a_removed_page_not_present_in_revision_b(tmp_path):
    source = tmp_path / "revision_b.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(source)
    delta = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=2,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Removed page content",
        confidence=1,
    )
    destination = write_markup(source, tmp_path / "marked.pdf", [delta])
    with pymupdf.open(destination) as marked:
        assert len(marked) == 1
        assert marked[0].first_annot is None


def test_markup_does_not_draw_removed_content_on_revision_b(tmp_path):
    source = tmp_path / "revision_b.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=200)
        document.save(source)

    removed = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=1,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Removed text: 'old note'",
        confidence=1,
    )

    destination = write_markup(source, tmp_path / "marked.pdf", [removed])

    with pymupdf.open(destination) as marked:
        assert marked[0].first_annot is None
