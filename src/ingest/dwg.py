"""Stub adapter for DWG files.

Kept behind the same FormatAdapter seam as the PDF adapters so a real DWG parser
or conversion pipeline can be added later without
touching the delta engine, report, or chat layers.
"""

from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter
from src.observability.logging import get_logger

logger = get_logger(__name__)


class DWGStubAdapter(FormatAdapter):
    """Recognizes .dwg files but does not parse them yet."""

    def supports(self, file_path: Path) -> bool:
        return Path(file_path).suffix.lower() == ".dwg"

    def parse(self, file_path: Path) -> CanonicalDocument:
        file_path = Path(file_path)
        logger.warning("dwg_ingestion_not_implemented", extra={"file": file_path.name})
        raise NotImplementedError(
            f"DWG ingestion for '{file_path.name}' is not implemented. "
            "This project supports native PDF and OCR'd scanned PDF end-to-end."
        )
