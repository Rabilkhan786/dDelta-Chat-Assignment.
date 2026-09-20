import fitz

from src.config.settings import project_path, settings
from src.ingest.pdf_native import NativePDFAdapter


def test_native_pdf_normalizes_text_into_lines() -> None:
    """Native extraction uses lines, matching the OCR adapter's canonical granularity."""
    pdf_path = project_path(settings.paths.revision_a)
    document = NativePDFAdapter().parse(pdf_path)
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            expected = [
                line
                for block in page.get_text("dict")["blocks"]
                for line in block.get("lines", [])
                if "".join(span["text"] for span in line.get("spans", [])).strip()
            ]
            assert len(document.pages[page_index].elements) == len(expected)
