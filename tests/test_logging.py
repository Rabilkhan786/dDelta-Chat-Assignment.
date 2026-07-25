from pathlib import Path

from src.ingest.pdf_native import NativePDFAdapter


def test_native_adapter_preserves_pdf_page_and_element_locations() -> None:
    """The supplied native sample should enter the canonical seam with locations."""
    sample = Path("data/input/Lift Gas compressor-P&ID.pdf")
    document = NativePDFAdapter().parse(sample)
    assert len(document.pages) == 1
    assert document.pages[0].elements
    assert all(element.bbox is not None for element in document.pages[0].elements)
