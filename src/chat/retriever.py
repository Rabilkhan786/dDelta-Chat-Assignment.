"""Retrieval over the three explicitly separate source collections."""

from src.chat.index import DocumentIndexer, IndexedDocument
from src.config.settings import settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


class HybridRetriever:
    """BM25 retriever whose results preserve source provenance and citations."""

    def __init__(self, indexer: DocumentIndexer | None = None) -> None:
        self.indexer = indexer or DocumentIndexer()

    def search(self, query: str, top_k: int | None = None) -> list[IndexedDocument]:
        """Return lexical matches, preferring one result from each relevant source."""
        if not query.strip():
            raise ValueError("Question must not be empty.")
        with stage(logger, "retrieval"):
            index, documents = self.indexer.load()
            scores = index.get_scores(self.indexer._tokenize(query))
            ranked = sorted(enumerate(scores), key=lambda item: (-item[1], documents[item[0]].element_id))
            limit = top_k or settings.retrieval.top_k
            matches = [documents[position] for position, score in ranked if score > 0][:limit]
        logger.info("retrieval_completed", extra={"hits": len(matches), "question_length": len(query)})
        return matches
