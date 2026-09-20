"""Native PDF, scanned PDF/OCR, classification, and DWG seam tests."""

from pathlib import Path

import pymupdf
import pytest

from src.config.settings import project_path, settings
from src.ingest.base import FormatAdapter
from src.ingest.classify import classify_text
from src.ingest.dwg import DWGStubAdapter
from src.ingest.lines import ocr_lines
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.ingest.registry import AutomaticPDFAdapter

# Ingestion

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


# Canonical text classification and line normalization


@pytest.mark.parametrize(
    ("text", "font_size", "expected"),
    [
        ("10 bar", None, "dimension"),
        ("NOTE 24: NEW VALVE", None, "note"),
        ("PSV-9066A", None, "symbol"),
        ("COMPRESSOR PLAN", 18, "title"),
        ("unstructured table-looking text", None, "text"),
    ],
)
def test_supported_text_classification(text, font_size, expected):
    assert classify_text(text, font_size).value == expected


def test_dwg_seam_is_real_but_explicitly_unimplemented():
    adapter = DWGStubAdapter()
    assert isinstance(adapter, FormatAdapter)
    assert adapter.supports(Path("drawing.DWG"))
    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.parse(Path("drawing.DWG"))


def test_ocr_words_are_grouped_sorted_and_scaled() -> None:
    data = {
        "text": ["VALVE", "OPEN", "", "ignored"],
        "conf": [90, 80, 99, -1],
        "block_num": [1, 1, 1, 2],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 1],
        "left": [20, 80, 0, 0],
        "top": [40, 40, 0, 0],
        "width": [40, 40, 0, 0],
        "height": [20, 20, 0, 0],
    }

    lines = ocr_lines(data, scale=2)

    assert len(lines) == 1
    assert lines[0]["text"] == "VALVE OPEN"
    assert lines[0]["confidence"] == pytest.approx(0.85)
    assert lines[0]["x0"] == 10.0
    assert lines[0]["y0"] == 20.0
    assert lines[0]["x1"] == 60.0
    assert lines[0]["y1"] == 30.0


def test_native_pdf_normalizes_text_into_lines() -> None:
    """Native extraction uses lines, matching OCR canonical granularity."""
    pdf_path = project_path(settings.paths.revision_a)
    document = NativePDFAdapter().parse(pdf_path)
    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            expected = [
                line
                for block in page.get_text("dict")["blocks"]
                for line in block.get("lines", [])
                if "".join(span["text"] for span in line.get("spans", [])).strip()
            ]
            assert len(document.pages[page_index].elements) == len(expected)


def test_native_adapter_reports_unsupported_and_missing_files(tmp_path) -> None:
    adapter = NativePDFAdapter()
    assert adapter.extracted_text_characters(tmp_path / "drawing.txt") == 0
    with pytest.raises(ValueError, match="does not support"):
        adapter.parse(tmp_path / "drawing.txt")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.parse(tmp_path / "missing.pdf")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.extracted_text_characters(tmp_path / "missing.pdf")
