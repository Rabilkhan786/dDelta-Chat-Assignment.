"""OCR adapter for scanned PDFs, isolated from all downstream logic."""

from hashlib import sha256
from pathlib import Path
from typing import Any

import fitz

from src.canonical.model import BoundingBox, CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.ingest.base import FormatAdapter
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class ScannedPDFAdapter(FormatAdapter):
    """Render PDF pages and extract OCR text into a CanonicalDocument."""

    def __init__(self, ocr_engine: Any | None = None, dpi: int = 200) -> None:
        self._ocr_engine = ocr_engine
        self.dpi = dpi

    def supports(self, file_path: Path) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    @property
    def ocr_engine(self) -> Any:
        """Lazily initialise PaddleOCR so native-PDF use has no OCR dependency cost."""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise RuntimeError("Scanned PDF ingestion requires the optional 'paddleocr' dependency.") from error
            self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
        return self._ocr_engine

    def parse(self, file_path: Path) -> CanonicalDocument:
        file_path = Path(file_path)
        if not self.supports(file_path):
            raise ValueError(f"ScannedPDFAdapter does not support '{file_path.suffix}'.")
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {file_path}")
        with stage(logger, "scanned_pdf_ingestion"):
            with fitz.open(file_path) as pdf:
                pages = [self._parse_page(pdf_page, page_number) for page_number, pdf_page in enumerate(pdf, start=1)]
        return CanonicalDocument(metadata=DocumentMetadata(document_id=sha256(file_path.read_bytes()).hexdigest(),
            pid=file_path.stem, file_name=file_path.name, file_type="pdf"), pages=pages)

    def _parse_page(self, pdf_page: fitz.Page, page_number: int) -> Page:
        scale = self.dpi / 72
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        results = self.ocr_engine.ocr(pixmap.tobytes("png"), cls=True) or []
        elements: list[Element] = []
        for index, result in enumerate(results[0] if results else [], start=1):
            points, recognition = result
            text, confidence = recognition
            text = str(text).strip()
            if not text:
                continue
            xs, ys = zip(*points)
            elements.append(Element(id=f"p{page_number}_ocr{index}", page_number=page_number, type=ElementType.TEXT,
                text=text, bbox=BoundingBox(x0=min(xs) / scale, y0=min(ys) / scale, x1=max(xs) / scale, y1=max(ys) / scale),
                source="ocr", ocr_confidence=float(confidence)))
        logger.info("ocr_page_extracted", extra={"page": page_number, "elements": len(elements)})
        return Page(page_number=page_number, width=pdf_page.rect.width, height=pdf_page.rect.height, elements=elements)
