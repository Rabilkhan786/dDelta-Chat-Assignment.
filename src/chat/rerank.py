"""Optional cross-encoder reranking for the post-RRF candidate list."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from src.config.settings import settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.chat.index import Excerpt


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Load the reranker once per CLI/API process."""
    return CrossEncoder(settings.reranker.model)


def rerank(query: str, candidates: list[Excerpt]) -> list[Excerpt]:
    """Order retrieved candidates by cross-encoder relevance."""
    if not candidates:
        return []

    with stage(logger, "cross_encoder_reranking"):
        scores = _model().predict([(query, item.text) for item in candidates])

    ordered = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [replace(item, score=round(float(score), 6)) for item, score in ordered]
