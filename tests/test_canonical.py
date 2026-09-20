"""Tests for the shared canonical representation and JSON seam."""

import json

import pytest
from pydantic import ValidationError

from src.canonical.model import (
    CanonicalDocument,
    DocumentMetadata,
    Element,
    ElementType,
    Page,
)
from src.canonical.serialization import write_canonical_document


def _document() -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(
            document_id="doc-1",
            pid="PID-1",
            file_name="drawing.pdf",
            file_type="pdf",
            revision="A",
        ),
        pages=[
            Page(
                page_number=1,
                width=100,
                height=200,
                elements=[
                    Element(
                        id="p1_l1",
                        page_number=1,
                        type=ElementType.NOTE,
                        text="NOTE 1",
                    )
                ],
            )
        ],
    )


def test_canonical_document_writes_readable_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "document.json"

    write_canonical_document(_document(), destination)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["metadata"]["revision"] == "A"
    assert saved["pages"][0]["elements"][0]["type"] == "note"


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"page_number": 0},
        {"ocr_confidence": 1.1},
    ],
)
def test_element_rejects_invalid_canonical_values(invalid_values) -> None:
    values = {
        "id": "bad",
        "page_number": 1,
        "type": ElementType.TEXT,
        "text": "value",
        **invalid_values,
    }
    with pytest.raises(ValidationError):
        Element(**values)
