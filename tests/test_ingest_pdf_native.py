import pymupdf
import pytest

from src.config.settings import project_path, settings
from src.ingest.pdf_native import NativePDFAdapter


def test_native_pdf_normalizes_text_into_lines() -> None:
    """Native extraction uses lines, matching the OCR adapter's canonical granularity."""
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
