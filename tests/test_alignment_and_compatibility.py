"""Focused tests for moved-element matching and revision compatibility warnings."""

from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Element,
    ElementType,
    Page,
)
from src.config.settings import project_path, settings
from src.delta.align import Aligner
from src.delta.compatibility import check_revision_compatibility
from src.delta.engine import DeltaEngine
from src.ingest.pdf_native import NativePDFAdapter


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


def test_unrelated_documents_get_a_visible_warning() -> None:
    old, new = _document("PSV-9066 PRESSURE", 10), _document("EXPORT COMPRESSOR", 10)
    result = check_revision_compatibility(old, new, minimum_similarity=0.5)
    assert result.compatible is False
    assert result.message


def test_supplied_unrelated_pair_warns_but_revision_pair_passes():
    adapter = NativePDFAdapter()
    base = adapter.parse(project_path(settings.paths.revision_a))
    revised = adapter.parse(project_path(settings.paths.revision_b))
    unrelated = adapter.parse(
        project_path("data/samples/different_systems/export_gas_compressor.pdf")
    )
    assert check_revision_compatibility(base, revised).compatible
    assert not check_revision_compatibility(base, unrelated).compatible
