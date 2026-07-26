"""Deterministic local retrieval index for both revisions and the delta report."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from nltk.corpus import stopwords

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
    
    """Build lexical and semantic indexes while retaining PID A/B/report provenance."""

    def __init__(
        self,
        index_path: Path | None = None,
        documents_path: Path | None = None,
        semantic_enabled: bool | None = None,
    ) -> None:
        self.index_path = index_path or project_path(settings.paths.bm25_index)
        self.documents_path = documents_path or project_path(settings.paths.bm25_documents)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.semantic_enabled = settings.retrieval.use_semantic if semantic_enabled is None else semantic_enabled
        self._vector_store: Any | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        
        """the text is preprocessed , remove stopword and return tokens"""
        
        STOP_WORDS = set(stopwords.words("english"))
        tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        return [token for token in tokens if token not in STOP_WORDS]

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
        
        """Persist BM25 and semantic indexes of PID A, PID B, and the delta report."""
        
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
            if self.semantic_enabled:
                self._build_semantic_index(documents)
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

    @staticmethod
    def document_key(document: IndexedDocument) -> str:
        
        """Return the stable identity used to join lexical and semantic results."""
        
        return f"{document.source}:{document.pid}:{document.page_number}:{document.element_id}"

    def semantic_search(self, query: str, limit: int) -> list[IndexedDocument]:
        """Search the persisted vector store and restore canonical citation metadata."""
        _, documents = self.load()
        documents_by_key = {self.document_key(document): document for document in documents}
        matches = self._semantic_store().similarity_search(query, k=limit)
        results: list[IndexedDocument] = []
        for match in matches:
            document = documents_by_key.get(match.metadata.get("document_key", ""))
            if document is not None:
                results.append(document)
        return results

    def _semantic_store(self) -> Any:
        
        """Create the configured persistent Chroma store only when semantic search is used."""
        
        if self._vector_store is None:
            try:
                from langchain_chroma import Chroma
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as error:
                raise RuntimeError(
                    "Semantic retrieval requires langchain-chroma and langchain-huggingface. "
                    "Run `uv sync --locked`."
                ) from error
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=settings.embedding.model,
                    model_kwargs={"device": settings.embedding.device},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as error:
                raise RuntimeError(
                    f"Unable to load semantic embedding model '{settings.embedding.model}'. "
                    "Check model availability, network access, and available memory."
                ) from error
            self._vector_store = Chroma(
                collection_name=settings.chroma.collection_name,
                persist_directory=str(project_path(settings.chroma.persist_directory)),
                embedding_function=embeddings,
            )
        return self._vector_store

    def _build_semantic_index(self, documents: list[IndexedDocument]) -> None:
        
        """Replace this project's semantic collection with the current canonical excerpts."""
        
        store = self._semantic_store()
        existing = store.get(include=[]).get("ids", [])
        if existing:
            store.delete(ids=existing)
        store.add_texts(
            texts=[document.text for document in documents],
            ids=[self.document_key(document) for document in documents],
            metadatas=[{"document_key": self.document_key(document)} for document in documents],
        )
        logger.info("semantic_index_built", extra={"documents": len(documents),
            "embedding_model": settings.embedding.model})
