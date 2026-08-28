"""Retrieval index over PID A, PID B, and the delta report, backed by Chroma.

Every canonical text element and every delta entry becomes one embedded excerpt.
Each excerpt keeps its source PID, page number, and element id as metadata so a
retrieved match can always be turned into a citation.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.canonical.model import CanonicalDocument, DeltaEntry
from src.config.settings import project_path, settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class Excerpt:
    """One retrievable, citable piece of evidence."""

    text: str
    source: str  # "pid_a", "pid_b", or "delta_report"
    pid: str
    page_number: int
    element_id: str


def _excerpt_id(excerpt: Excerpt) -> str:
    return f"{excerpt.source}:{excerpt.pid}:{excerpt.page_number}:{excerpt.element_id}"


def _vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.model, model_kwargs={"device": settings.embedding.device})
    return Chroma(
        collection_name=settings.chroma.collection_name,
        persist_directory=str(project_path(settings.chroma.persist_directory)),
        embedding_function=embeddings,
    )


def _document_excerpts(document: CanonicalDocument, source: str) -> list[Excerpt]:
    return [
        Excerpt(element.text.strip(), source, document.metadata.pid, page.page_number, element.id)
        for page in document.pages for element in page.elements if element.text.strip()
    ]


def _delta_excerpts(deltas: list[DeltaEntry], pid: str) -> list[Excerpt]:
    return [
        Excerpt(f"{delta.change_type.value} {delta.element_type.value}: {delta.description}",
            "delta_report", pid, delta.page_number, f"delta-{index}")
        for index, delta in enumerate(deltas, start=1)
    ]


def build_index(pid_a: CanonicalDocument, pid_b: CanonicalDocument, deltas: list[DeltaEntry]) -> int:
    """Embed PID A, PID B, and the delta report into a fresh Chroma collection."""
    excerpts = _document_excerpts(pid_a, "pid_a") + _document_excerpts(pid_b, "pid_b") + _delta_excerpts(deltas, pid_b.metadata.pid)
    if not excerpts:
        raise ValueError("Cannot build a retrieval index from empty canonical documents.")

    with stage(logger, "retrieval_index_build"):
        store = _vector_store()
        existing_ids = store.get(include=[]).get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_texts(
            texts=[excerpt.text for excerpt in excerpts],
            ids=[_excerpt_id(excerpt) for excerpt in excerpts],
            metadatas=[{"source": excerpt.source, "pid": excerpt.pid, "page_number": excerpt.page_number,
                "element_id": excerpt.element_id} for excerpt in excerpts],
        )
    logger.info("retrieval_index_built", extra={"excerpts": len(excerpts)})
    return len(excerpts)


def search(query: str, top_k: int | None = None) -> list[Excerpt]:
    """Return the top matching excerpts for a question, ranked by semantic similarity."""
    if not query.strip():
        raise ValueError("Question must not be empty.")
    with stage(logger, "retrieval"):
        matches = _vector_store().similarity_search(query, k=top_k or settings.retrieval.top_k)
    results = [Excerpt(match.page_content, match.metadata["source"], match.metadata["pid"],
        match.metadata["page_number"], match.metadata["element_id"]) for match in matches]
    logger.info("retrieval_completed", extra={"hits": len(results), "question_length": len(query)})
    return results
