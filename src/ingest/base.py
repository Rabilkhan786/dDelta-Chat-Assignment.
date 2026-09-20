from abc import ABC, abstractmethod
from pathlib import Path

from src.canonical.model import CanonicalDocument


class FormatAdapter(ABC):
    """
    Base interface for all document ingestion adapters.
    Every supported document format must implement this interface.
    """

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """
        Return True if this adapter can process the given file.
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self, file_path: Path) -> CanonicalDocument:
        """
        Parse the input document and return a CanonicalDocument.
        """
        raise NotImplementedError
