"""Native (text-layer) PDF adapter."""

from hashlib import sha256
from pathlib import Path

import fitz

from src.canonical.model import BoundingBox, CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.ingest.base import FormatAdapter
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

        text_dict = pdf_page.get_text("dict")

        for block_index, block in enumerate(text_dict["blocks"], start=1):

            # Skip non-text blocks (images, drawings, etc.)
            if block.get("type") != 0:
                continue

            for line_index, line in enumerate(block.get("lines", []), start=1):

                spans = line.get("spans", [])

                if not spans:
                    continue

               # Merge all spans in the line into one text string
                text = "".join(span["text"] for span in spans).strip()
                text = " ".join(text.split())

            

                if not text:
                    continue

                # Compute bounding box for the entire line
                x0 = min(span["bbox"][0] for span in spans)
                y0 = min(span["bbox"][1] for span in spans)
                x1 = max(span["bbox"][2] for span in spans)
                y1 = max(span["bbox"][3] for span in spans)

                elements.append(
                    Element(
                        id=f"p{page_number}_b{block_index}_l{line_index}",
                        page_number=page_number,
                        type=ElementType.TEXT,
                        text=text,
                        bbox=BoundingBox(
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        ),
                    )
                )

        logger.info(
            "page_extracted",
            extra={
                "page": page_number,
                "elements": len(elements),
            },
        )

        return Page(
            page_number=page_number,
            width=pdf_page.rect.width,
            height=pdf_page.rect.height,
            elements=elements,
        )