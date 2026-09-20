from pathlib import Path

import pytest

from src.config.settings import project_path, settings
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.ingest.registry import AutomaticPDFAdapter

NATIVE_SAMPLE = project_path(settings.paths.revision_a)
SCANNED_SAMPLE = Path("data/samples/scanned/lift_gas_scanned.pdf")


def test_native_pdf_detection_uses_selectable_text() -> None:
    """A PDF with a real text layer must route to the native adapter."""
    router = AutomaticPDFAdapter(text_threshold=30)
    assert isinstance(router.select_adapter(NATIVE_SAMPLE), NativePDFAdapter)


def test_scanned_pdf_detection_routes_to_ocr() -> None:
    """A rasterized, text-free PDF must route to the OCR adapter."""
    router = AutomaticPDFAdapter(text_threshold=30)
    assert isinstance(router.select_adapter(SCANNED_SAMPLE), ScannedPDFAdapter)


def test_scanned_pdf_ocr_recovers_text_with_bounding_boxes() -> None:
    """Tesseract must recover real text, source, confidence, and location from a scan."""
    document = ScannedPDFAdapter().parse(SCANNED_SAMPLE)
    elements = document.pages[0].elements
    assert elements
    assert all(element.source == "ocr" for element in elements)
    assert all(element.bbox is not None for element in elements)
    assert all(0.0 <= element.ocr_confidence <= 1.0 for element in elements)
    assert any("COMPRESSOR" in element.text.upper() for element in elements)


def test_pdf_router_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        AutomaticPDFAdapter(text_threshold=0)


def test_ocr_adapter_rejects_unusable_dpi() -> None:
    with pytest.raises(ValueError, match="at least 72"):
        ScannedPDFAdapter(dpi=50)


def test_ocr_adapter_reports_unsupported_and_missing_files(tmp_path) -> None:
    adapter = ScannedPDFAdapter()
    with pytest.raises(ValueError, match="does not support"):
        adapter.parse(tmp_path / "drawing.txt")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.parse(tmp_path / "missing.pdf")


def test_pdf_router_rejects_non_pdf_input(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not support"):
        AutomaticPDFAdapter().select_adapter(tmp_path / "drawing.dwg")
