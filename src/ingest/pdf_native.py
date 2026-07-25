from pathlib import Path
import uuid
import fitz

from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Element,
    ElementType,
    Page,
)
from src.ingest.base import FormatAdapter
from src.observability.logging import setup_logger

logger = setup_logger(__name__)


class NativePDFAdapter(FormatAdapter):
    """
    Adapter for machine-readable PDF documents.
    Converts a native PDF into the canonical representation.
    """

    SUPPORTED_SUFFIXES = {".pdf"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def parse(self, file_path: Path) -> CanonicalDocument:

        logger.info(
            "Starting native PDF ingestion",
            extra={"file": str(file_path)}
        )

        if not self.supports(file_path):
            logger.error(
                "Unsupported file type",
                extra={"file": str(file_path)}
            )
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        pdf = fitz.open(file_path)

        try:

            pages = []

            for page_index, pdf_page in enumerate(pdf):

                logger.info(
                    "Extracting page",
                    extra={"page": page_index + 1}
                )

                elements = []

                blocks = pdf_page.get_text("blocks")

                for block_index, block in enumerate(blocks):

                    x0, y0, x1, y1, text, *_ = block

                    text = text.strip()

                    if not text:
                        continue

                    elements.append(
                        Element(
                            id=f"p{page_index + 1}_b{block_index + 1}",
                            page_number=page_index + 1,
                            type=ElementType.TEXT,
                            text=text,
                            bbox=BoundingBox(
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                            )
                        )
                    )

                pages.append(
                    Page(
                        page_number=page_index + 1,
                        width=pdf_page.rect.width,
                        height=pdf_page.rect.height,
                        elements=elements,
                    )
                )

                logger.info(
                    "Page extracted successfully",
                    extra={
                        "page": page_index + 1,
                        "elements": len(elements),
                    },
                )

            metadata = DocumentMetadata(
                document_id=str(uuid.uuid4()),
                pid=file_path.stem,
                file_name=file_path.name,
                file_type="pdf",
                revision=None,
            )

            document = CanonicalDocument(
                metadata=metadata,
                pages=pages,
            )

            logger.info(
                "Native PDF ingestion completed",
                extra={
                    "pages": len(pages),
                    "file": file_path.name,
                },
            )

            return document

        except Exception:

            logger.exception(
                "Native PDF ingestion failed",
                extra={"file": str(file_path)},
            )
            raise

        finally:
            pdf.close()
            logger.info(
                "PDF file closed",
                extra={"file": str(file_path)}
            )