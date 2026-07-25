"""Adapter selection at the ingestion boundary."""

from pathlib import Path
from typing import Iterable

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter


class AdapterRegistry:
    """Select exactly one registered adapter for a source document."""

    def __init__(self, adapters: Iterable[FormatAdapter]) -> None:
        self._adapters = list(adapters)

    def parse(self, file_path: Path) -> CanonicalDocument:
        """Parse through the selected adapter, never leaking format downstream."""
        matches = [adapter for adapter in self._adapters if adapter.supports(file_path)]
        if not matches:
            raise ValueError(f"No ingestion adapter supports '{file_path.suffix}'.")
        if len(matches) > 1:
            raise ValueError("More than one adapter supports this source; select the intended adapter explicitly.")
        return matches[0].parse(file_path)
