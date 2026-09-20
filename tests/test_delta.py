"""Revision compatibility, alignment, delta, and report tests."""

import json

from src.canonical.model import (
    Alignment,
    AlignmentResult,
    BoundingBox,
    CanonicalDocument,
    DeltaEntry,
    DeltaType,
    DocumentMetadata,
    Element,
    ElementType,
    Page,
)
from src.config.settings import project_path, settings
from src.delta.align import Aligner
from src.delta.compatibility import CompatibilityResult, check_revision_compatibility
from src.delta.engine import DeltaEngine
from src.delta.report import DeltaReportGenerator
from src.ingest.pdf_native import NativePDFAdapter

# Delta Support


def _alignment_document(text: str, x: float) -> CanonicalDocument:
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
    old, new = _alignment_document("PSV-9066", 10), _alignment_document("PSV-9066", 300)
    alignment = Aligner(max_bbox_distance=50).align(old, new)
    delta = DeltaEngine().compare(alignment)[0]
    assert delta.change_type.value == "moved"
    assert delta.location_changed is True


def test_missing_bounding_boxes_do_not_create_a_false_move() -> None:
    old = _alignment_document("PSV-9066", 10)
    new = _alignment_document("PSV-9066", 10)
    old.pages[0].elements[0].bbox = None
    new.pages[0].elements[0].bbox = None

    alignment = Aligner().align(old, new)
    delta = DeltaEngine().compare(alignment)[0]

    assert delta.change_type.value == "unchanged"
    assert delta.location_changed is False


# Delta Compatibility


def _compatibility_document(text: str) -> CanonicalDocument:
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
        _compatibility_document("PSV-9066 PRESSURE"),
        _compatibility_document("EXPORT COMPRESSOR"),
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


# Delta Engine


def test_modified_native_elements_have_normalized_confidence() -> None:
    """RapidFuzz's 0-100 score must become a report confidence from 0 to 1."""
    old = Element(
        id="a",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure 10",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    new = Element(
        id="b",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure 12",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    delta = DeltaEngine().compare(
        AlignmentResult(matches=[Alignment(left=old, right=new, similarity=90, bbox_distance=0)])
    )[0]
    assert delta.change_type.value == "modified"
    assert delta.confidence == 0.9


def test_rewrapped_whitespace_is_not_reported_as_modified() -> None:
    """Re-flowed line breaks/spacing alone shouldn't count as a real content change."""
    old = Element(
        id="a",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure  10\nbar",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    new = Element(
        id="b",
        page_number=1,
        type=ElementType.TEXT,
        text="pressure 10 bar",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )
    delta = DeltaEngine().compare(
        AlignmentResult(matches=[Alignment(left=old, right=new, similarity=100, bbox_distance=0)])
    )[0]
    assert delta.change_type.value == "unchanged"


def test_ocr_confidence_discounts_similarity() -> None:
    """An OCR match should be less confident than an identical native-text match."""
    old = Element(
        id="a", page_number=1, type=ElementType.TEXT, text="valve", source="ocr", ocr_confidence=0.5
    )
    new = Element(
        id="b", page_number=1, type=ElementType.TEXT, text="valve", source="ocr", ocr_confidence=0.5
    )
    delta = DeltaEngine().compare(
        AlignmentResult(matches=[Alignment(left=old, right=new, similarity=100, bbox_distance=0)])
    )[0]
    assert delta.change_type.value == "unchanged"
    assert delta.confidence == 0.5


def test_low_ocr_confidence_in_revision_a_is_not_lost_when_b_is_native():
    old = Element(
        id="a",
        page_number=1,
        type=ElementType.TEXT,
        text="10 bar",
        source="ocr",
        ocr_confidence=0.2,
    )
    new = Element(id="b", page_number=1, type=ElementType.TEXT, text="12 bar")
    match = Alignment(left=old, right=new, similarity=90, bbox_distance=0)
    delta = DeltaEngine().compare(AlignmentResult(matches=[match]))[0]
    assert delta.confidence == 0.18


def test_removed_labels_far_above_each_other_are_not_joined():
    lower = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=1,
        description="Removed text: 'lower'",
        confidence=1,
        region=BoundingBox(x0=10, y0=200, x1=50, y1=210),
    )
    upper = lower.model_copy(update={"region": BoundingBox(x0=10, y0=10, x1=50, y1=20)})
    assert not DeltaEngine._can_join(lower, upper)


def test_adjacent_removed_fragments_are_joined_into_one_callout() -> None:
    first = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=1,
        description="Removed text: 'MECHANICAL'",
        confidence=0.9,
        region=BoundingBox(x0=10, y0=10, x1=60, y1=20),
    )
    second = first.model_copy(
        update={
            "description": "Removed text: 'INTERLOCK'",
            "confidence": 0.8,
            "region": BoundingBox(x0=10, y0=21, x1=60, y1=31),
        }
    )

    joined = DeltaEngine._join_adjacent_fragments([first, second])

    assert len(joined) == 1
    assert joined[0].description == "Removed text: 'MECHANICAL INTERLOCK'"
    assert joined[0].confidence == 0.8


def test_removed_element_id_belongs_to_previous_revision() -> None:
    old = Element(
        id="old-id",
        page_number=1,
        type=ElementType.TEXT,
        text="removed note",
    )

    delta = DeltaEngine().compare(AlignmentResult(unmatched_left=[old]))[0]

    assert delta.element_id is None
    assert delta.previous_element_id == "old-id"


# Delta Report


def _report_document(name: str, revision: str) -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(
            document_id=revision,
            pid=name,
            file_name=f"{name}.pdf",
            file_type="pdf",
            revision=revision,
        )
    )


def test_report_writes_only_meaningful_changes_with_location(tmp_path) -> None:
    unchanged = DeltaEntry(
        change_type=DeltaType.UNCHANGED,
        element_type=ElementType.TEXT,
        page_number=1,
        description="Unchanged text: 'old'",
        confidence=1,
    )
    added = DeltaEntry(
        change_type=DeltaType.ADDED,
        element_type=ElementType.NOTE,
        page_number=2,
        region=BoundingBox(x0=1, y0=2, x1=3, y1=4),
        description="Added note: 'new'",
        confidence=0.9,
        element_id="p2_l1",
    )
    json_path = tmp_path / "delta.json"
    markdown_path = tmp_path / "delta.md"

    report = DeltaReportGenerator(json_path, markdown_path).generate(
        _report_document("old", "A"),
        _report_document("new", "B"),
        [unchanged, added],
        CompatibilityResult(0.4, False, "Review this pair."),
    )

    assert report["summary"]["actual_changes"] == 1
    assert report["entries"][0]["location_revision"] == "B"
    assert report["entries"][0]["bounding_box"]["x0"] == 1
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Compatibility Warning" in markdown
    assert "delta-1" in markdown


def test_report_explains_when_no_changes_exist(tmp_path) -> None:
    markdown_path = tmp_path / "delta.md"
    report = DeltaReportGenerator(tmp_path / "delta.json", markdown_path).generate(
        _report_document("old", "A"),
        _report_document("new", "B"),
        [],
    )

    assert report["entries"] == []
    assert "No meaningful changes" in markdown_path.read_text(encoding="utf-8")
