from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter
from src.observability.logging import get_logger

logger = get_logger(__name__)


class DWGStubAdapter(FormatAdapter):
    """
    Stub adapter for DWG files.

    This adapter exists to satisfy the project architecture until
    a real DWG parser (ezdxf, ODA, Autodesk APIs, etc.) is added.
    """

    SUPPORTED_EXTENSIONS = {".dwg"}

    @classmethod
    def supports(cls, file_path: str | Path) -> bool:
        """
        Returns True if the file has a .dwg extension.
        """
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def parse(self, file_path: str | Path) -> CanonicalDocument:
        """
        Stub implementation.
        """
        file_path = Path(file_path)

        logger.warning(
            "DWG ingestion requested for '%s', but this adapter is currently a stub.",
            file_path.name,
        )

        raise NotImplementedError(
            "DWG ingestion is not implemented yet. "
            "This project currently supports Native PDF. "
            "Scanned PDF support will be added separately."
        )