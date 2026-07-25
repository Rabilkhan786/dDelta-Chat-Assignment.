"""Deterministic local retrieval index for both revisions and the delta report."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.canonical.model import CanonicalDocument, DeltaEntry
from src.config.settings import project_path, settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexedDocument:
    """A retrievable excerpt with stable metadata for answer citations."""

    text: str
    source: str
    pid: str
    page_number: int
    element_id: str
    bounding_box: dict[str, float] | None


class DocumentIndexer:
    """Build and persist a BM25 index while retaining PID A/B/report provenance."""

    def __init__(self, index_path: Path | None = None, documents_path: Path | None = None) -> None:
        self.index_path = index_path or project_path(settings.paths.bm25_index)
        self.documents_path = documents_path or project_path(settings.paths.bm25_documents)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())

    def create_documents(self, document: CanonicalDocument, source: str) -> list[IndexedDocument]:
        """Convert one canonical revision into citation-ready retrieval excerpts."""
        result: list[IndexedDocument] = []
        for page in document.pages:
            for element in page.elements:
                if not element.text.strip():
                    continue
                bbox = element.bbox.model_dump() if element.bbox else None
                result.append(IndexedDocument(element.text.strip(), source, document.metadata.pid,
                    page.page_number, element.id, bbox))
        return result

    def create_delta_documents(self, deltas: list[DeltaEntry], pid: str) -> list[IndexedDocument]:
        """Convert the deterministic delta output into its own retrieval source."""
        return [IndexedDocument(
            text=f"{delta.change_type.value} {delta.element_type.value}: {delta.description}",
            source="delta_report", pid=pid, page_number=delta.page_number,
            element_id=f"delta-{index}",
            bounding_box=delta.region.model_dump() if delta.region else None,
        ) for index, delta in enumerate(deltas, start=1)]

    def build(self, pid_a: CanonicalDocument, pid_b: CanonicalDocument, deltas: list[DeltaEntry]) -> int:
        """Persist a reproducible index of PID A, PID B, and the delta report."""
        with stage(logger, "retrieval_index_build"):
            documents = (self.create_documents(pid_a, "pid_a") + self.create_documents(pid_b, "pid_b")
                         + self.create_delta_documents(deltas, pid_b.metadata.pid))
            if not documents:
                raise ValueError("Cannot build retrieval index from empty canonical documents.")
            corpus = [self._tokenize(document.text) for document in documents]
            with self.documents_path.open("wb") as stream:
                pickle.dump(documents, stream)
            with self.index_path.open("wb") as stream:
                pickle.dump(BM25Okapi(corpus), stream)
        logger.info("retrieval_index_built", extra={"documents": len(documents)})
        return len(documents)

    def load(self) -> tuple[BM25Okapi, list[IndexedDocument]]:
        """Load an existing index or raise an actionable setup error."""
        if not self.index_path.exists() or not self.documents_path.exists():
            raise FileNotFoundError("Retrieval index is missing. Run `python main.py run` first.")
        with self.index_path.open("rb") as stream:
            index = pickle.load(stream)
        with self.documents_path.open("rb") as stream:
            documents = pickle.load(stream)
        return index, documents
