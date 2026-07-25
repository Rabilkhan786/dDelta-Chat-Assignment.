from pathlib import Path

from src.canonical.model import CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.chat.answer import GroundedChatService
from src.chat.index import DocumentIndexer
from src.chat.llm import LLMResponse
from src.chat.retriever import HybridRetriever


class FakeProvider:
    """Provider test double proving chat only receives retrieved evidence."""

    def __init__(self) -> None:
        self.prompt = ""

    def complete(self, prompt: str) -> LLMResponse:
        self.prompt = prompt
        return LLMResponse("Supported answer [pid_a | PID A | page 1 | a-1]", 1, 1, 0.0)


def _document(pid: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(metadata=DocumentMetadata(document_id=pid, pid=pid, file_name=f"{pid}.pdf", file_type="pdf"),
        pages=[Page(page_number=1, width=100, height=100,
            elements=[Element(id=f"{pid.lower()}-1", page_number=1, type=ElementType.TEXT, text=text)])])


def test_grounded_chat_uses_retrieved_evidence_and_returns_citations(tmp_path: Path) -> None:
    """A provider never receives raw documents, only retrieved citation-labelled excerpts."""
    indexer = DocumentIndexer(tmp_path / "index.pkl", tmp_path / "documents.pkl")
    indexer.build(_document("A", "pump pressure is 10 bar"), _document("B", "pump pressure is 12 bar"), [])
    provider = FakeProvider()
    answer = GroundedChatService(HybridRetriever(indexer), provider).answer("What is the pump pressure?")
    assert answer.citations
    assert "PID A" in provider.prompt
    assert answer.text.startswith("Supported answer")
