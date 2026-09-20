"""Native (text-layer) PDF adapter."""

from hashlib import sha256
from pathlib import Path

import fitz

from src.canonical.model import BoundingBox, CanonicalDocument, DocumentMetadata, Element, Page
from src.ingest.base import FormatAdapter
from src.ingest.classify import classify_text
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class NativePDFAdapter(FormatAdapter):
    """Convert a machine-readable PDF into the canonical representation."""

    SUPPORTED_SUFFIXES = {".pdf"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def extracted_text_characters(self, file_path: Path) -> int:
        """Count selectable, non-whitespace text for automatic PDF routing."""
        file_path = Path(file_path)
        if not self.supports(file_path):
            return 0
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {file_path}")
        try:
            with fitz.open(file_path) as pdf:
                return sum(len(page.get_text("text").strip()) for page in pdf)
        except fitz.FileDataError as error:
            raise ValueError(f"Unable to open PDF '{file_path.name}'.") from error

    def parse(self, file_path: Path) -> CanonicalDocument:
        file_path = Path(file_path)
        if not self.supports(file_path):
            raise ValueError(f"NativePDFAdapter does not support '{file_path.suffix}'.")
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {file_path}")

        with stage(logger, "native_pdf_ingestion"):
            try:
                with fitz.open(file_path) as pdf:
                    pages = [self._parse_page(pdf_page, page_number) for page_number, pdf_page in enumerate(pdf, start=1)]
            except fitz.FileDataError as error:
                raise ValueError(f"Unable to open PDF '{file_path.name}'.") from error

        return CanonicalDocument(
            metadata=DocumentMetadata(
                document_id=sha256(file_path.read_bytes()).hexdigest(),
                pid=file_path.stem,
                file_name=file_path.name,
                file_type="pdf",
            ),
            pages=pages,
        )

    @staticmethod
    def _parse_page(pdf_page: fitz.Page, page_number: int) -> Page:
        elements: list[Element] = []
        text = pdf_page.get_text("dict")
        line_index = 0
        for block in text["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                content = "".join(span["text"] for span in spans).strip()
                if not content:
                    continue
                line_index += 1
                x0, y0, x1, y1 = line["bbox"]
                font_size = max((span["size"] for span in spans), default=None)
                elements.append(Element(
                    id=f"p{page_number}_l{line_index}", page_number=page_number,
                    type=classify_text(content, font_size), text=content,
                    bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                ))
        logger.info("page_extracted", extra={"page": page_number, "elements": len(elements)})
        return Page(page_number=page_number, width=pdf_page.rect.width, height=pdf_page.rect.height, elements=elements)
