"""Detect whether a PDF is native or scanned, then route it to the right adapter."""

from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.config.settings import settings
from src.ingest.base import FormatAdapter
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class AutomaticPDFAdapter(FormatAdapter):
    """Count selectable text to decide between the native and OCR adapters."""

    def __init__(
        self,
        native_adapter: NativePDFAdapter | None = None,
        ocr_adapter: ScannedPDFAdapter | None = None,
        text_threshold: int | None = None,
    ) -> None:
        self.native_adapter = native_adapter or NativePDFAdapter()
        self.ocr_adapter = ocr_adapter or ScannedPDFAdapter()
        self.text_threshold = text_threshold or settings.ingest.native_text_threshold

    def supports(self, file_path: Path) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def select_adapter(self, file_path: Path) -> FormatAdapter:
        """Native when selectable text is meaningful, otherwise OCR."""
        file_path = Path(file_path)
        if not self.supports(file_path):
            raise ValueError(f"AutomaticPDFAdapter does not support '{file_path.suffix}'.")
        text_characters = self.native_adapter.extracted_text_characters(file_path)
        adapter: FormatAdapter = self.native_adapter if text_characters >= self.text_threshold else self.ocr_adapter
        logger.info("pdf_adapter_selected", extra={"adapter": type(adapter).__name__,
            "selectable_text_characters": text_characters, "text_threshold": self.text_threshold, "file": file_path.name})
        return adapter

    def parse(self, file_path: Path) -> CanonicalDocument:
        with stage(logger, "pdf_type_detection"):
            return self.select_adapter(file_path).parse(Path(file_path))
