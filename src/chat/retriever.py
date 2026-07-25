"""Hybrid retrieval over the three explicitly separate source collections."""

from src.chat.index import DocumentIndexer, IndexedDocument
from src.config.settings import settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class HybridRetriever:
    """Fuse lexical BM25 and semantic vector rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        indexer: DocumentIndexer | None = None,
        use_bm25: bool | None = None,
        use_semantic: bool | None = None,
        hybrid: bool | None = None,
    ) -> None:
        self.indexer = indexer or DocumentIndexer()
        self.use_bm25 = settings.retrieval.use_bm25 if use_bm25 is None else use_bm25
        self.use_semantic = settings.retrieval.use_semantic if use_semantic is None else use_semantic
        self.hybrid = settings.retrieval.hybrid if hybrid is None else hybrid

    def search(self, query: str, top_k: int | None = None) -> list[IndexedDocument]:
        """Return BM25 and semantic results fused by RRF while preserving citations."""
        if not query.strip():
            raise ValueError("Question must not be empty.")
        with stage(logger, "retrieval"):
            limit = top_k or settings.retrieval.top_k
            ranked_results: list[list[IndexedDocument]] = []
            if self.use_bm25:
                ranked_results.append(self._bm25_search(query, limit))
            if self.use_semantic:
                ranked_results.append(self.indexer.semantic_search(query, limit))
            if not ranked_results:
                raise RuntimeError("Both lexical and semantic retrieval are disabled in configuration.")
            matches = self.reciprocal_rank_fusion(ranked_results, limit) if self.hybrid else ranked_results[0]
        logger.info("retrieval_completed", extra={"hits": len(matches), "question_length": len(query),
            "bm25_enabled": self.use_bm25, "semantic_enabled": self.use_semantic,
            "fusion": "rrf" if self.hybrid else "disabled"})
        return matches

    def _bm25_search(self, query: str, limit: int) -> list[IndexedDocument]:
        """Rank lexical matches, including common terms whose BM25 score may be zero."""
        index, documents = self.indexer.load()
        query_tokens = self.indexer._tokenize(query)
        scores = index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], documents[item[0]].element_id))
        return [
            documents[position]
            for position, _ in ranked
            if set(query_tokens) & set(self.indexer._tokenize(documents[position].text))
        ][:limit]

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[IndexedDocument]],
        limit: int,
        rrf_k: int | None = None,
    ) -> list[IndexedDocument]:
        """Merge ranked retrieval lists deterministically using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        documents: dict[str, IndexedDocument] = {}
        constant = rrf_k or settings.retrieval.rrf_k
        for ranked_list in ranked_lists:
            for rank, document in enumerate(ranked_list, start=1):
                key = DocumentIndexer.document_key(document)
                documents[key] = document
                scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
        ordered_keys = sorted(scores, key=lambda key: (-scores[key], key))
        return [documents[key] for key in ordered_keys[:limit]]
