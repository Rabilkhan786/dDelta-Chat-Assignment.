from pathlib import Path

import fitz

from src.ingest.pdf_native import NativePDFAdapter


def test_every_non_empty_native_pdf_block_becomes_a_canonical_element() -> None:
    """Verify native ingestion preserves each extractable source-text block."""
    pdf_path = Path("data/input/Lift Gas compressor-P&ID.pdf")
    document = NativePDFAdapter().parse(pdf_path)
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            expected = [block for block in page.get_text("blocks") if block[4].strip()]
            assert len(document.pages[page_index].elements) == len(expected)
