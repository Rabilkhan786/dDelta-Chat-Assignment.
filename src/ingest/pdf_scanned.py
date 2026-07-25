from pathlib import Path
import uuid

import fitz
from paddleocr import PaddleOCR

from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Element,
    ElementType,
    Page,
)

from src.ingest.base import FormatAdapter
from src.observability.logging import get_logger
from src.observability.tracing import span


logger = get_logger(__name__)


class ScannedPDFAdapter(FormatAdapter):

    def __init__(self):

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en"
        )
    
    def supports(self, file_path: Path) -> bool:

        return file_path.suffix.lower() == ".pdf"
    
    def parse(self, file_path: Path) -> CanonicalDocument:

        logger.info("Starting scanned PDF ingestion")

        pages = []

        pdf = fitz.open(file_path)
        
        metadata = DocumentMetadata(
            pid=file_path.stem,
            file_name=file_path.name,
            file_type="pdf",
            revision=file_path.name
        )
        
        