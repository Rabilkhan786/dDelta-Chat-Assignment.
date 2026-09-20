"""Tests for revision compatibility warnings."""

from src.canonical.model import CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.config.settings import project_path, settings
from src.delta.compatibility import check_revision_compatibility
from src.ingest.pdf_native import NativePDFAdapter


def _document(text: str) -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(
            document_id=text,
            pid="sample",
            file_name="sample.pdf",
            file_type="pdf",
        ),
        pages=[
            Page(
                page_number=1,
                width=100,
                height=100,
                elements=[
                    Element(
                        id="p1_l1",
                        page_number=1,
                        type=ElementType.TEXT,
                        text=text,
                    )
                ],
            )
        ],
    )


def test_unrelated_documents_get_a_visible_warning() -> None:
    result = check_revision_compatibility(
        _document("PSV-9066 PRESSURE"),
        _document("EXPORT COMPRESSOR"),
        minimum_similarity=0.5,
    )
    assert result.compatible is False
    assert result.message


def test_supplied_unrelated_pair_warns_but_revision_pair_passes() -> None:
    adapter = NativePDFAdapter()
    base = adapter.parse(project_path(settings.paths.revision_a))
    revised = adapter.parse(project_path(settings.paths.revision_b))
    unrelated = adapter.parse(
        project_path("data/samples/different_systems/export_gas_compressor.pdf")
    )
    assert check_revision_compatibility(base, revised).compatible
    assert not check_revision_compatibility(base, unrelated).compatible
