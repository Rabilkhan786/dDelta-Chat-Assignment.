"""Small hybrid retrieval over both revisions and the deterministic delta report.

BM25 keeps exact drawing tags and dimensions reliable. Chroma adds semantic
matches. Reciprocal Rank Fusion combines their rankings before optional reranking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Plus

from src.canonical.model import CanonicalDocument, DeltaEntry
from src.chat.query import keyword_tokens, route_question, search_queries
from src.config.settings import project_path, settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


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


def _excerpt_id(excerpt: Excerpt) -> str:
    return f"{excerpt.source}:{excerpt.pid}:{excerpt.page_number}:{excerpt.element_id}"


@lru_cache(maxsize=1)
def _embeddings(model: str, device: str) -> HuggingFaceEmbeddings:
    """Reuse loaded weights across requests in the same CLI/API process."""
    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": device})


def _vector_store() -> Chroma:
    embeddings = _embeddings(settings.embedding.model, settings.embedding.device)
    return Chroma(
        collection_name=settings.chroma.collection_name,
        persist_directory=str(project_path(settings.chroma.persist_directory)),
        embedding_function=embeddings,
    )


def _document_excerpts(document: CanonicalDocument, source: str) -> list[Excerpt]:
    return [
        Excerpt(
            element.text.strip(),
            source,
            document.metadata.pid,
            page.page_number,
            element.id,
            document.metadata.revision,
            element.type.value,
            bbox=json.dumps(element.bbox.model_dump()) if element.bbox else None,
            confidence=element.ocr_confidence,
        )
        for page in document.pages
        for element in page.elements
        if element.text.strip()
    ]


def _delta_excerpts(deltas: list[DeltaEntry], pid: str, revision: str | None) -> list[Excerpt]:
    return [
        Excerpt(
            f"{delta.change_type.value} {delta.element_type.value}: {delta.description}",
            "delta_report",
            pid,
            delta.page_number,
            f"delta-{index}",
            revision,
            delta.element_type.value,
            delta.change_type.value,
            json.dumps(delta.region.model_dump()) if delta.region else None,
            delta.confidence,
        )
        for index, delta in enumerate(deltas, start=1)
    ]


def build_index(
    pid_a: CanonicalDocument, pid_b: CanonicalDocument, deltas: list[DeltaEntry]
) -> int:
    """Build local vector storage; BM25 is rebuilt from its small content at query time."""
    excerpts = (
        _document_excerpts(pid_a, "pid_a")
        + _document_excerpts(pid_b, "pid_b")
        + _delta_excerpts(deltas, pid_b.metadata.pid, pid_b.metadata.revision)
    )
    if not excerpts:
        raise ValueError("Cannot build a retrieval index from empty canonical documents.")
    with stage(logger, "retrieval_index_build"):
        store = _vector_store()
        existing_ids = store.get(include=[]).get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_texts(
            texts=[item.text for item in excerpts],
            ids=[_excerpt_id(item) for item in excerpts],
            metadatas=[_metadata(item) for item in excerpts],
        )
    logger.info("retrieval_index_built", extra={"excerpts": len(excerpts)})
    return len(excerpts)


def search(query: str, top_k: int | None = None) -> list[Excerpt]:
    """Prepare query, retrieve, fuse, rerank, and return up to top_k excerpts."""
    queries = search_queries(query, rewrite=settings.retrieval.rewrite_query)
    result_count = settings.retrieval.top_k if top_k is None else top_k
    if result_count < 1:
        raise ValueError("top_k must be positive.")
    candidate_count = max(result_count, settings.retrieval.candidate_k)
    with stage(logger, "hybrid_retrieval"):
        store = _vector_store()
        raw = store.get(include=["documents", "metadatas"])
        all_items = [
            _from_values(text, values)
            for text, values in zip(raw.get("documents", []) or [], raw.get("metadatas", []) or [])
        ]
        if not all_items:
            logger.info("retrieval_completed", extra={"hits": 0, "reason": "empty_index"})
            return []
        lexical_scores = _keyword_scores(query, all_items)
        semantic_scores = _semantic_scores(store, queries, min(candidate_count, len(all_items)))
        preferred_sources = route_question(query)
        candidates = _fuse_candidates(
            all_items, lexical_scores, semantic_scores, preferred_sources
        )[:candidate_count]
        if settings.reranker.enabled:
            from src.chat.rerank import rerank

            candidates = rerank(queries[0], candidates)
        results = candidates[:result_count]
        logger.info(
            "retrieval_completed",
            extra={
                "hits": len(results),
                "query_variants": queries,
                "route": sorted(preferred_sources) if preferred_sources else "all",
                "reranker_enabled": settings.reranker.enabled,
            },
        )
        return results


def _keyword_scores(query: str, items: list[Excerpt]) -> dict[str, float]:
    """BM25+ keeps common exact tags useful even in tiny two-document corpora."""
    corpus = [keyword_tokens(item.text) for item in items]
    tokens = list(dict.fromkeys(keyword_tokens(query)))
    if not tokens or not any(corpus):
        return {}
    scores = BM25Plus(corpus).get_scores(tokens)
    # BM25+ has a baseline score even without overlap. Only real matches count.
    return {
        _excerpt_id(item): float(score)
        for item, words, score in zip(items, corpus, scores)
        if set(tokens).intersection(words)
    }


def _semantic_scores(store: Chroma, queries: list[str], count: int) -> dict[str, float]:
    """Keep each excerpt's best distance across variants; do not double-count it."""
    scores: dict[str, float] = {}
    for query in queries:
        for document, distance in store.similarity_search_with_score(query, k=count):
            item_id = _excerpt_id(_from_values(document.page_content, document.metadata))
            score = 1 / (1 + max(float(distance), 0))
            if score >= settings.retrieval.minimum_vector_similarity:
                scores[item_id] = max(scores.get(item_id, 0), score)
    return scores


def _fuse_candidates(
    items: list[Excerpt],
    lexical: dict[str, float],
    semantic: dict[str, float],
    preferred_sources: set[str] | None,
) -> list[Excerpt]:
    """Fuse one ranking per retrieval method, then apply the source preference."""
    rankings = [
        sorted(scores, key=lambda key: (-scores[key], key)) for scores in (lexical, semantic)
    ]
    scores = reciprocal_rank_fusion(rankings)
    candidates = []
    for item in items:
        item_id = _excerpt_id(item)
        if item_id not in scores:
            continue
        boost = (
            settings.retrieval.source_boost
            if preferred_sources and item.source in preferred_sources
            else 1.0
        )
        candidates.append(replace(item, score=scores[item_id] * boost))
    return sorted(candidates, key=lambda item: (-(item.score or 0), _excerpt_id(item)))


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int | None = None) -> dict[str, float]:
    """Combine ranked IDs; an item supported by both methods rises to the top."""
    denominator = rrf_k or settings.retrieval.rrf_k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[item_id] = scores.get(item_id, 0) + 1 / (denominator + rank)
    return scores


def _metadata(item: Excerpt) -> dict[str, str | int | float | None]:
    values = asdict(item)
    values.pop("text")
    values.pop("score")
    return {key: value for key, value in values.items() if value is not None}


def _from_values(text: str, values: dict) -> Excerpt:
    return Excerpt(
        text,
        values["source"],
        values["pid"],
        int(values["page_number"]),
        values["element_id"],
        values.get("revision"),
        values.get("element_type"),
        values.get("change_type"),
        values.get("bbox"),
        values.get("confidence"),
    )
