import pytest

from src.chat import index
from src.chat.answer import GroundedChatService
from src.chat.index import Excerpt
from src.chat.llm import LLMResponse


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


def _stub_search(monkeypatch: pytest.MonkeyPatch, results: list[Excerpt]) -> None:
    monkeypatch.setattr(index, "search", lambda query, top_k=None: results)


def test_grounded_chat_uses_retrieved_evidence_and_returns_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider never receives raw documents, only retrieved citation-labelled excerpts."""
    _stub_search(monkeypatch, [Excerpt("pump pressure is 10 bar", "pid_a", "A", 1, "a-1")])
    provider = FakeProvider()
    answer = GroundedChatService(provider).answer("What is the pump pressure?")
    assert answer.citations
    assert "PID A" in provider.prompt
    assert answer.text.startswith("Supported answer")


def test_grounded_chat_refuses_when_nothing_is_retrieved(monkeypatch: pytest.MonkeyPatch) -> None:
    """No supporting evidence must produce an explicit refusal, never a guess."""
    _stub_search(monkeypatch, [])
    answer = GroundedChatService(FakeProvider()).answer("anything")
    assert answer.citations == []
    assert "cannot support" in answer.text


def test_grounded_chat_returns_evidence_when_provider_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable provider must not become an unsupported fabricated answer."""
    _stub_search(monkeypatch, [Excerpt("pump pressure is 10 bar", "pid_a", "A", 1, "a-1")])
    answer = GroundedChatService(FailingProvider()).answer("What is the pump pressure?")
    assert answer.citations
    assert "could not complete" in answer.text


def test_grounded_chat_returns_evidence_when_provider_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials must not turn retrieved evidence into an uncaught error."""
    _stub_search(monkeypatch, [Excerpt("pump pressure is 10 bar", "pid_a", "A", 1, "a-1")])
    monkeypatch.setattr("src.chat.answer.configured_provider", lambda: (_ for _ in ()).throw(RuntimeError("missing key")))
    answer = GroundedChatService().answer("What is the pump pressure?")
    assert answer.citations
    assert "could not complete" in answer.text
