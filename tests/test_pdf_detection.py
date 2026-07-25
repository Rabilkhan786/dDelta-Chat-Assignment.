from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.canonical.model import CanonicalDocument, DocumentMetadata
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.ingest.registry import AutomaticPDFAdapter


SAMPLE = Path("data/input/Lift Gas compressor-P&ID.pdf")


def test_native_pdf_detection_uses_selectable_text() -> None:
    """The supplied native P&ID must route to the native adapter automatically."""
    router = AutomaticPDFAdapter(text_threshold=30)
    assert isinstance(router.select_adapter(SAMPLE), NativePDFAdapter)


def test_ocr_pdf_detection_routes_when_selectable_text_is_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PDF with no meaningful text must route to OCR before parsing begins."""
    native = NativePDFAdapter()
    ocr = ScannedPDFAdapter(ocr_engine=object())
    monkeypatch.setattr(native, "extracted_text_characters", lambda _: 0)
    router = AutomaticPDFAdapter(native_adapter=native, ocr_adapter=ocr, text_threshold=30)
    assert router.select_adapter(SAMPLE) is ocr


def test_router_returns_the_selected_adapter_canonical_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing preserves the canonical representation used by downstream modules."""
    native = NativePDFAdapter()
    expected = CanonicalDocument(metadata=DocumentMetadata(document_id="ocr", pid="ocr", file_name="scan.pdf", file_type="pdf"))

    class CanonicalOCRAdapter(ScannedPDFAdapter):
        def parse(self, file_path: Path) -> CanonicalDocument:
            return expected

    ocr = CanonicalOCRAdapter(ocr_engine=object())
    monkeypatch.setattr(native, "extracted_text_characters", lambda _: 0)
    router = AutomaticPDFAdapter(native_adapter=native, ocr_adapter=ocr, text_threshold=30)
    assert router.parse(SAMPLE) == expected


def test_ocr_missing_paddleocr_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional OCR dependencies must fail clearly without fabricated output."""
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(RuntimeError, match="Install it with"):
        _ = ScannedPDFAdapter().ocr_engine


@pytest.mark.skip(reason="No scanned PDF sample was supplied; add one to validate real PaddleOCR end-to-end ingestion.")
def test_scanned_pdf_end_to_end_placeholder() -> None:
    """Documented placeholder: real OCR validation requires a supplied scanned PDF."""
