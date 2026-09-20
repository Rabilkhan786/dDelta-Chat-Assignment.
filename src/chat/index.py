"""Small hybrid retrieval over both revisions and the deterministic delta report.

BM25 keeps exact drawing tags and dimensions reliable. Chroma adds semantic
matches. Reciprocal Rank Fusion combines their rankings without a second model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.canonical.model import CanonicalDocument, DeltaEntry
from src.config.settings import project_path, settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-/]\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class Excerpt:
    """One citable result with enough metadata for stable citations."""

    text: str
    source: str
    pid: str
    page_number: int
    element_id: str
    revision: str | None = None
    element_type: str | None = None
    change_type: str | None = None
    bbox: str | None = None
    confidence: float | None = None
    score: float | None = None


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _excerpt_id(excerpt: Excerpt) -> str:
    return f"{excerpt.source}:{excerpt.pid}:{excerpt.page_number}:{excerpt.element_id}"


def _vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.model,
        model_kwargs={"device": settings.embedding.device})
    return Chroma(collection_name=settings.chroma.collection_name,
        persist_directory=str(project_path(settings.chroma.persist_directory)), embedding_function=embeddings)


def _document_excerpts(document: CanonicalDocument, source: str) -> list[Excerpt]:
    return [Excerpt(element.text.strip(), source, document.metadata.pid, page.page_number,
        element.id, document.metadata.revision, element.type.value,
        bbox=json.dumps(element.bbox.model_dump()) if element.bbox else None,
        confidence=element.ocr_confidence)
        for page in document.pages for element in page.elements if element.text.strip()]


def _delta_excerpts(deltas: list[DeltaEntry], pid: str, revision: str | None) -> list[Excerpt]:
    return [Excerpt(f"{delta.change_type.value} {delta.element_type.value}: {delta.description}",
        "delta_report", pid, delta.page_number, f"delta-{index}", revision,
        delta.element_type.value, delta.change_type.value,
        json.dumps(delta.region.model_dump()) if delta.region else None, delta.confidence)
        for index, delta in enumerate(deltas, start=1)]


def build_index(pid_a: CanonicalDocument, pid_b: CanonicalDocument, deltas: list[DeltaEntry]) -> int:
    """Build local vector storage; BM25 is rebuilt from its small content at query time."""
    excerpts = (_document_excerpts(pid_a, "pid_a") + _document_excerpts(pid_b, "pid_b")
        + _delta_excerpts(deltas, pid_b.metadata.pid, pid_b.metadata.revision))
    if not excerpts:
        raise ValueError("Cannot build a retrieval index from empty canonical documents.")
    with stage(logger, "retrieval_index_build"):
        store = _vector_store()
        existing_ids = store.get(include=[]).get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_texts(texts=[item.text for item in excerpts], ids=[_excerpt_id(item) for item in excerpts],
            metadatas=[_metadata(item) for item in excerpts])
    logger.info("retrieval_index_built", extra={"excerpts": len(excerpts)})
    return len(excerpts)


def route_question(question: str) -> set[str] | None:
    """Prefer a narrow source only for plainly scoped questions."""
    lowered = question.lower()
    if any(word in lowered for word in ("compare", "difference", "between revision")):
        return None
    if any(word in lowered for word in ("changed", "change", "added", "removed", "modified", "moved")):
        return {"delta_report"}
    if any(text in lowered for text in ("revision a", "pid a", "old revision", "base revision")):
        return {"pid_a"}
    if any(text in lowered for text in ("revision b", "pid b", "new revision", "revised")):
        return {"pid_b"}
    return None


def search(query: str, top_k: int | None = None) -> list[Excerpt]:
    """Fuse exact-keyword and semantic matches, rejecting weak evidence."""
    if not query.strip():
        raise ValueError("Question must not be empty.")
    result_count = top_k or settings.retrieval.top_k
    preferred_sources = route_question(query)
    with stage(logger, "hybrid_retrieval"):
        store = _vector_store()
        raw = store.get(include=["documents", "metadatas"])
        all_items = [_from_values(text, values) for text, values in zip(raw.get("documents", []) or [], raw.get("metadatas", []) or [])]
        if not all_items:
            return []
        bm25_scores = BM25Okapi([_tokens(item.text) for item in all_items]).get_scores(_tokens(query))
        keyword_ranks = [index for index, score in sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True) if score > 0]
        vector_matches = store.similarity_search_with_score(query, k=min(settings.retrieval.candidate_k, len(all_items)))
        vector_scores = {_excerpt_id(_from_document(document)): 1 / (1 + max(distance, 0))
            for document, distance in vector_matches}

    fused = reciprocal_rank_fusion([
        [_excerpt_id(all_items[index]) for index in keyword_ranks],
        sorted(vector_scores, key=vector_scores.get, reverse=True),
    ])

    by_id = {_excerpt_id(item): item for item in all_items}
    bm25_by_id = {_excerpt_id(item): score for item, score in zip(all_items, bm25_scores)}
    results: list[Excerpt] = []
    for item_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
        item = by_id[item_id]
        if bm25_by_id.get(item_id, 0) <= 0 and vector_scores.get(item_id, 0) < settings.retrieval.minimum_vector_similarity:
            continue
        boost = 1.25 if preferred_sources and item.source in preferred_sources else 1.0
        results.append(_with_score(item, score * boost))
    results.sort(key=lambda item: item.score or 0, reverse=True)
    results = results[:settings.retrieval.candidate_k]
    if settings.reranker.enabled:
        # Import here to keep the cross-encoder model out of indexing and startup.
        from src.chat.rerank import rerank
        results = rerank(query, results)
    results = results[:result_count]
    logger.info("retrieval_completed", extra={"hits": len(results), "question_length": len(query),
        "route": sorted(preferred_sources) if preferred_sources else "all",
        "reranker_enabled": settings.reranker.enabled})
    return results


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int | None = None) -> dict[str, float]:
    """Combine ranked IDs; an item supported by both methods rises to the top."""
    denominator = rrf_k or settings.retrieval.rrf_k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0) + 1 / (denominator + rank)
    return scores


def _metadata(item: Excerpt) -> dict[str, str | int | float | None]:
    values = {"source": item.source, "pid": item.pid, "page_number": item.page_number,
        "element_id": item.element_id, "revision": item.revision, "element_type": item.element_type,
        "change_type": item.change_type, "bbox": item.bbox, "confidence": item.confidence}
    return {key: value for key, value in values.items() if value is not None}


def _from_document(document: object) -> Excerpt:
    return _from_values(document.page_content, document.metadata)


def _from_values(text: str, values: dict) -> Excerpt:
    return Excerpt(text, values["source"], values["pid"], int(values["page_number"]), values["element_id"],
        values.get("revision"), values.get("element_type"), values.get("change_type"), values.get("bbox"), values.get("confidence"))


def _with_score(item: Excerpt, score: float) -> Excerpt:
    return Excerpt(item.text, item.source, item.pid, item.page_number, item.element_id, item.revision,
        item.element_type, item.change_type, item.bbox, item.confidence, round(score, 6))
