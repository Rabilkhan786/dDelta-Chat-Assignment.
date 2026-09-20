"""Hybrid retrieval over PID A, PID B, and the generated delta report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Plus

from src.canonical.model import CanonicalDocument
from src.chat.query import keyword_tokens
from src.config.settings import project_path, settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class Excerpt:
    """One citable retrieval result."""

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
    """Reuse embedding weights within the same CLI/API process."""
    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": device})


def _vector_store() -> Chroma:
    embeddings = _embeddings(settings.embedding.model, settings.embedding.device)
    return Chroma(
        collection_name=settings.chroma.collection_name,
        persist_directory=str(project_path(settings.chroma.persist_directory)),
        embedding_function=embeddings,
    )


def _document_excerpts(document: CanonicalDocument, source: str) -> list[Excerpt]:
    """Create searchable excerpts from one canonical PID revision."""
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


def _delta_excerpts(report: dict, pid: str, revision: str | None) -> list[Excerpt]:
    """Create a report summary plus one searchable excerpt per actual change."""
    entries = report.get("entries", [])
    excerpts: list[Excerpt] = []

    if entries:
        summary = report.get("summary", {})
        actual_changes = int(summary.get("actual_changes", len(entries)))
        modified = int(summary.get("modified", 0))
        removed = int(summary.get("removed", 0))
        added = int(summary.get("added", 0))
        moved = int(summary.get("moved", 0))
        excerpts.append(
            Excerpt(
                "Delta report summary. "
                f"{actual_changes} changes detected: "
                f"{modified} modified, {removed} removed, "
                f"{added} added, {moved} moved.",
                "delta_report",
                pid,
                1,
                "delta-summary",
                revision,
                change_type="summary",
            )
        )

    for entry in entries:
        bbox = entry.get("bounding_box")
        excerpts.append(
            Excerpt(
                f"Delta report change. {entry['change_type']}: {entry['description']}",
                "delta_report",
                pid,
                int(entry["page_number"]),
                entry["delta_id"],
                revision,
                entry.get("element_type"),
                entry.get("change_type"),
                json.dumps(bbox) if bbox else None,
                float(entry["confidence"]) if entry.get("confidence") is not None else None,
            )
        )

    return excerpts


def build_index(
    pid_a: CanonicalDocument,
    pid_b: CanonicalDocument,
    report: dict,
) -> int:
    """Build the retrieval index from both PIDs and the generated delta report."""
    excerpts = (
        _document_excerpts(pid_a, "pid_a")
        + _document_excerpts(pid_b, "pid_b")
        + _delta_excerpts(report, pid_b.metadata.pid, pid_b.metadata.revision)
    )
    if not excerpts:
        raise ValueError("Cannot build a retrieval index from empty documents.")

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
    """Search all indexed evidence with BM25, vectors, RRF, and reranking."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    result_count = settings.retrieval.top_k if top_k is None else top_k
    if result_count < 1:
        raise ValueError("top_k must be positive.")

    candidate_count = max(result_count, settings.retrieval.candidate_k)

    with stage(logger, "hybrid_retrieval"):
        store = _vector_store()
        raw = store.get(include=["documents", "metadatas"])
        all_items = [
            _from_values(text, values)
            for text, values in zip(
                raw.get("documents", []) or [],
                raw.get("metadatas", []) or [],
                strict=True,
            )
        ]

        if not all_items:
            logger.info(
                "retrieval_completed",
                extra={"hits": 0, "query": query, "reason": "empty_index"},
            )
            return []

        lexical_scores = _keyword_scores(query, all_items)
        semantic_scores = _semantic_scores(
            store,
            query,
            min(candidate_count, len(all_items)),
        )
        fused_candidates = _fuse_candidates(
            all_items,
            lexical_scores,
            semantic_scores,
        )
        candidates = _balanced_candidate_pool(
            all_items,
            fused_candidates,
            lexical_scores,
            semantic_scores,
            candidate_count,
        )
        if settings.reranker.enabled:
            from src.chat.rerank import rerank

            candidates = rerank(query, candidates)

        results = candidates[:result_count]
        logger.info(
            "retrieval_completed",
            extra={
                "hits": len(results),
                "query": query,
                "keyword_hits": len(lexical_scores),
                "semantic_hits": len(semantic_scores),
                "fused_candidates": len(fused_candidates),
                "reranked_candidates": len(candidates),
                "reranker_enabled": settings.reranker.enabled,
            },
        )
        return results


def _keyword_scores(query: str, items: list[Excerpt]) -> dict[str, float]:
    """Score exact lexical overlap with BM25+."""
    corpus = [keyword_tokens(item.text) for item in items]
    tokens = keyword_tokens(query)
    if not tokens or not any(corpus):
        return {}

    scores = BM25Plus(corpus).get_scores(tokens)
    return {
        _excerpt_id(item): float(score)
        for item, words, score in zip(items, corpus, scores, strict=True)
        if set(tokens).intersection(words)
    }


def _semantic_scores(store: Chroma, query: str, count: int) -> dict[str, float]:
    """Convert Chroma distances into bounded similarity scores."""
    scores: dict[str, float] = {}
    for document, distance in store.similarity_search_with_score(query, k=count):
        item_id = _excerpt_id(_from_values(document.page_content, document.metadata))
        score = 1 / (1 + max(float(distance), 0))
        if score >= settings.retrieval.minimum_vector_similarity:
            scores[item_id] = score
    return scores


def _fuse_candidates(
    items: list[Excerpt],
    lexical: dict[str, float],
    semantic: dict[str, float],
) -> list[Excerpt]:
    """Fuse BM25 and semantic rankings with Reciprocal Rank Fusion."""
    rankings = [
        sorted(scores, key=lambda key: (-scores[key], key)) for scores in (lexical, semantic)
    ]
    scores = reciprocal_rank_fusion(rankings)

    candidates = [
        replace(item, score=scores[_excerpt_id(item)])
        for item in items
        if _excerpt_id(item) in scores
    ]
    return sorted(
        candidates,
        key=lambda item: (-(item.score or 0), _excerpt_id(item)),
    )


def _balanced_candidate_pool(
    items: list[Excerpt],
    fused: list[Excerpt],
    lexical: dict[str, float],
    semantic: dict[str, float],
    count: int,
) -> list[Excerpt]:
    """Reserve candidates from both retrievers and the small delta report."""
    item_by_id = {_excerpt_id(item): item for item in items}
    fused_ids = [_excerpt_id(item) for item in fused[:count]]
    lexical_ids = sorted(lexical, key=lambda key: (-lexical[key], key))[:count]
    semantic_ids = sorted(semantic, key=lambda key: (-semantic[key], key))[:count]
    delta_ids = [_excerpt_id(item) for item in items if item.source == "delta_report"]
    if len(delta_ids) > count:
        delta_ids = []
    selected_ids = list(dict.fromkeys(fused_ids + lexical_ids + semantic_ids + delta_ids))
    return [item_by_id[item_id] for item_id in selected_ids]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    rrf_k: int | None = None,
) -> dict[str, float]:
    """Combine rankings so evidence found by both retrievers rises."""
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
