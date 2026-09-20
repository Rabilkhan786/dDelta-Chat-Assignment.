"""Focused tests for deterministic nearby and moved-element alignment."""

from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Element,
    ElementType,
    Page,
)
from src.delta.align import Aligner
from src.delta.engine import DeltaEngine


def _document(text: str, x: float) -> CanonicalDocument:
    element = Element(
        id=f"id-{x}",
        page_number=1,
        type=ElementType.SYMBOL,
        text=text,
        bbox=BoundingBox(x0=x, y0=10, x1=x + 10, y1=20),
    )
    return CanonicalDocument(
        metadata=DocumentMetadata(
            document_id=str(x), pid="sample", file_name="sample.pdf", file_type="pdf"
        ),
        pages=[Page(page_number=1, width=500, height=500, elements=[element])],
    )


def test_far_identical_tag_is_reported_as_moved() -> None:
    old, new = _document("PSV-9066", 10), _document("PSV-9066", 300)
    alignment = Aligner(max_bbox_distance=50).align(old, new)
    delta = DeltaEngine().compare(alignment)[0]
    assert delta.change_type.value == "moved"
    assert delta.location_changed is True


def test_missing_bounding_boxes_do_not_create_a_false_move() -> None:
    old = _document("PSV-9066", 10)
    new = _document("PSV-9066", 10)
    old.pages[0].elements[0].bbox = None
    new.pages[0].elements[0].bbox = None

    alignment = Aligner().align(old, new)
    delta = DeltaEngine().compare(alignment)[0]

    assert delta.change_type.value == "unchanged"
    assert delta.location_changed is False
