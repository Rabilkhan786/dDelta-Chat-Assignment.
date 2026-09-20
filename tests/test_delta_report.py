"""Tests for machine-readable and human-readable delta reports."""

import json

from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DeltaEntry,
    DeltaType,
    DocumentMetadata,
    ElementType,
)
from src.delta.compatibility import CompatibilityResult
from src.delta.report import DeltaReportGenerator


def _document(name: str, revision: str) -> CanonicalDocument:
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
        _document("old", "A"),
        _document("new", "B"),
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
        _document("old", "A"),
        _document("new", "B"),
        [],
    )

    assert report["entries"] == []
    assert "No meaningful changes" in markdown_path.read_text(encoding="utf-8")
