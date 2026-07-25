from pathlib import Path

from src.canonical.model import CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.chat.answer import GroundedChatService
from src.chat.index import DocumentIndexer, IndexedDocument
from src.chat.llm import LLMResponse
from src.chat.retriever import HybridRetriever


class FakeProvider:
    """Provider test double proving chat only receives retrieved evidence."""

    def __init__(self) -> None:
        self.prompt = ""

    def complete(self, prompt: str) -> LLMResponse:
        self.prompt = prompt
        return LLMResponse("Supported answer [pid_a | PID A | page 1 | a-1]", 1, 1, 0.0)


class FailingProvider:
    """Provider test double for observable, non-fabricated provider failures."""

    def complete(self, prompt: str) -> LLMResponse:
        raise RuntimeError("provider unavailable")


def _document(pid: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(metadata=DocumentMetadata(document_id=pid, pid=pid, file_name=f"{pid}.pdf", file_type="pdf"),
        pages=[Page(page_number=1, width=100, height=100,
            elements=[Element(id=f"{pid.lower()}-1", page_number=1, type=ElementType.TEXT, text=text)])])


def test_grounded_chat_uses_retrieved_evidence_and_returns_citations(tmp_path: Path) -> None:
    """A provider never receives raw documents, only retrieved citation-labelled excerpts."""
    indexer = DocumentIndexer(tmp_path / "index.pkl", tmp_path / "documents.pkl", semantic_enabled=False)
    indexer.build(_document("A", "pump pressure is 10 bar"), _document("B", "pump pressure is 12 bar"), [])
    provider = FakeProvider()
    answer = GroundedChatService(HybridRetriever(indexer, use_semantic=False), provider).answer("What is the pump pressure?")
    assert answer.citations
    assert "PID A" in provider.prompt
    assert answer.text.startswith("Supported answer")


def test_grounded_chat_returns_evidence_when_provider_fails(tmp_path: Path) -> None:
    """An unavailable provider must not become an unsupported fabricated answer."""
    indexer = DocumentIndexer(tmp_path / "index.pkl", tmp_path / "documents.pkl", semantic_enabled=False)
    indexer.build(_document("A", "pump pressure is 10 bar"), _document("B", "pump pressure is 12 bar"), [])
    answer = GroundedChatService(HybridRetriever(indexer, use_semantic=False), FailingProvider()).answer("What is the pump pressure?")
    assert answer.citations
    assert "could not complete" in answer.text


def test_reciprocal_rank_fusion_rewards_results_found_by_both_retrievers() -> None:
    """RRF must promote an excerpt returned by both BM25 and semantic search."""
    first = IndexedDocument("first", "pid_a", "A", 1, "a-1", None)
    shared = IndexedDocument("shared", "pid_b", "B", 1, "b-1", None)
    semantic_only = IndexedDocument("semantic", "delta_report", "B", 1, "delta-1", None)
    fused = HybridRetriever.reciprocal_rank_fusion([[first, shared], [shared, semantic_only]], limit=3, rrf_k=60)
    assert fused[0] == shared


def test_hybrid_retriever_fuses_bm25_and_semantic_rankings(tmp_path: Path, monkeypatch) -> None:
    """Hybrid search keeps results contributed by each independently ranked retriever."""
    indexer = DocumentIndexer(tmp_path / "index.pkl", tmp_path / "documents.pkl", semantic_enabled=False)
    indexer.build(_document("A", "pump pressure is 10 bar"), _document("B", "compressor alarm is active"), [])
    _, documents = indexer.load()
    monkeypatch.setattr(indexer, "semantic_search", lambda query, limit: [documents[1]])

    results = HybridRetriever(indexer, use_semantic=True, hybrid=True).search("pump pressure", top_k=2)

    assert {result.source for result in results} == {"pid_a", "pid_b"}
