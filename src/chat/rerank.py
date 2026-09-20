"""Optional small cross-encoder reranker for the post-RRF candidate list."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from src.config.settings import settings

if TYPE_CHECKING:
    from src.chat.index import Excerpt


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Load once, only when a query needs reranking."""
    return CrossEncoder(settings.reranker.model)


def rerank(query: str, candidates: list[Excerpt]) -> list[Excerpt]:
    """Return the most query-relevant RRF candidates in cross-encoder order."""
    if not candidates:
        return []
    scores = _model().predict([(query, item.text) for item in candidates])
    ordered = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
    return [_with_rerank_score(item, float(score)) for item, score in ordered[:settings.reranker.top_k]]


def _with_rerank_score(item: Excerpt, score: float) -> Excerpt:
    """Keep the public retrieval shape while replacing its ranking score."""
    return type(item)(item.text, item.source, item.pid, item.page_number, item.element_id,
        item.revision, item.element_type, item.change_type, item.bbox, item.confidence, round(score, 6))
