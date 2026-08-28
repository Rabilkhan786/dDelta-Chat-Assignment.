"""OCR adapter for scanned PDFs, using Tesseract to recover text and layout."""

import io
from hashlib import sha256
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from src.canonical.model import BoundingBox, CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.config.settings import settings
from src.ingest.base import FormatAdapter
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class ScannedPDFAdapter(FormatAdapter):
    """Render each page to an image and OCR it with Tesseract into the canonical model."""

    def __init__(self, dpi: int | None = None) -> None:
        self.dpi = dpi or settings.ingest.ocr_dpi

    def supports(self, file_path: Path) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> CanonicalDocument:
        file_path = Path(file_path)
        if not self.supports(file_path):
            raise ValueError(f"ScannedPDFAdapter does not support '{file_path.suffix}'.")
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {file_path}")

        with stage(logger, "scanned_pdf_ingestion"):
            with fitz.open(file_path) as pdf:
                pages = [self._parse_page(pdf_page, page_number) for page_number, pdf_page in enumerate(pdf, start=1)]

        return CanonicalDocument(
            metadata=DocumentMetadata(
                document_id=sha256(file_path.read_bytes()).hexdigest(),
                pid=file_path.stem,
                file_name=file_path.name,
                file_type="pdf",
            ),
            pages=pages,
        )

    def _parse_page(self, pdf_page: fitz.Page, page_number: int) -> Page:
        """Render one page to an image at the configured DPI and OCR it."""
        scale = self.dpi / 72
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        elements: list[Element] = []
        for index, text in enumerate(ocr_data["text"]):
            text = text.strip()
            confidence = float(ocr_data["conf"][index])
            if not text or confidence < 0:
                continue
            left, top, width, height = (ocr_data[key][index] for key in ("left", "top", "width", "height"))
            elements.append(Element(
                id=f"p{page_number}_ocr{index}",
                page_number=page_number,
                type=ElementType.TEXT,
                text=text,
                bbox=BoundingBox(x0=left / scale, y0=top / scale, x1=(left + width) / scale, y1=(top + height) / scale),
                source="ocr",
                ocr_confidence=round(confidence / 100, 2),
            ))

        logger.info("ocr_page_extracted", extra={"page": page_number, "elements": len(elements)})
        return Page(page_number=page_number, width=pdf_page.rect.width, height=pdf_page.rect.height, elements=elements)
