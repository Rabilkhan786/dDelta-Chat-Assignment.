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


def test_grounded_chat_uses_retrieved_evidence_and_returns_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_grounded_chat_returns_evidence_when_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable provider must not become an unsupported fabricated answer."""
    _stub_search(monkeypatch, [Excerpt("pump pressure is 10 bar", "pid_a", "A", 1, "a-1")])
    answer = GroundedChatService(FailingProvider()).answer("What is the pump pressure?")
    assert answer.citations
    assert "could not complete" in answer.text


def test_grounded_chat_returns_evidence_when_provider_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing credentials must not turn retrieved evidence into an uncaught error."""
    _stub_search(monkeypatch, [Excerpt("pump pressure is 10 bar", "pid_a", "A", 1, "a-1")])
    monkeypatch.setattr(
        "src.chat.answer.configured_provider",
        lambda: (_ for _ in ()).throw(RuntimeError("missing key")),
    )
    answer = GroundedChatService().answer("What is the pump pressure?")
    assert answer.citations
    assert "could not complete" in answer.text


@pytest.mark.parametrize("text", ["Pressure is 10 bar.", "Pressure is 10 bar [invented source]."])
def test_chat_rejects_missing_or_unknown_citations(monkeypatch, text):
    class Provider:
        def complete(self, prompt):
            return LLMResponse(text, 1, 1, 0.0)

    _stub_search(monkeypatch, [Excerpt("10 bar", "pid_a", "A", 1, "a-1")])
    result = GroundedChatService(Provider()).answer("pressure?")
    assert result.status == "unsupported"
    assert result.citations == []


def test_chat_returns_only_citations_actually_used(monkeypatch):
    _stub_search(
        monkeypatch,
        [Excerpt("10 bar", "pid_a", "A", 1, "a-1"), Excerpt("12 bar", "pid_b", "B", 1, "b-1")],
    )
    result = GroundedChatService(FakeProvider()).answer("pressure?")
    assert result.citations == ["[pid_a | PID A | page 1 | a-1]"]


def test_chat_preserves_shared_request_id(monkeypatch):
    _stub_search(monkeypatch, [])
    result = GroundedChatService().answer("anything", request_id="pipeline-request")
    assert result.request_id == "pipeline-request"


def test_chat_accepts_unicode_bracket_variant_for_known_citation(monkeypatch):
    class UnicodeBracketProvider:
        def complete(self, prompt):
            return LLMResponse(
                "Pressure is 10 bar 【pid_a | PID A | page 1 | a-1]",
                1,
                1,
                0.0,
            )

    _stub_search(monkeypatch, [Excerpt("10 bar", "pid_a", "A", 1, "a-1")])
    result = GroundedChatService(UnicodeBracketProvider()).answer("pressure?")

    assert result.status == "answered"
    assert result.citations == ["[pid_a | PID A | page 1 | a-1]"]
    assert "【" not in result.text
    assert "[pid_a | PID A | page 1 | a-1]" in result.text


def test_chat_rejects_unicode_bracket_variant_for_unknown_citation(monkeypatch):
    class UnknownCitationProvider:
        def complete(self, prompt):
            return LLMResponse(
                "Pressure is 10 bar 【invented source】",
                1,
                1,
                0.0,
            )

    _stub_search(monkeypatch, [Excerpt("10 bar", "pid_a", "A", 1, "a-1")])
    result = GroundedChatService(UnknownCitationProvider()).answer("pressure?")

    assert result.status == "unsupported"
    assert result.citations == []


def test_chat_normalizes_unicode_spaces_inside_known_citation(monkeypatch):
    class UnicodeSpaceProvider:
        def complete(self, prompt):
            return LLMResponse(
                "Pressure is 10 bar [pid_a | PID A | page\u202f1 | a-1]",
                1,
                1,
                0.0,
            )

    _stub_search(monkeypatch, [Excerpt("10 bar", "pid_a", "A", 1, "a-1")])
    result = GroundedChatService(UnicodeSpaceProvider()).answer("pressure?")

    assert result.status == "answered"
    assert result.text == "Pressure is 10 bar [pid_a | PID A | page 1 | a-1]"


def test_chat_allows_bracketed_prose_with_an_exact_citation(monkeypatch):
    class BracketedProseProvider:
        def complete(self, prompt):
            return LLMResponse(
                "Added [NOTE 24] [pid_a | PID A | page 1 | a-1]",
                1,
                1,
                0.0,
            )

    _stub_search(monkeypatch, [Excerpt("NOTE 24", "pid_a", "A", 1, "a-1")])
    result = GroundedChatService(BracketedProseProvider()).answer("note?")

    assert result.status == "answered"
    assert result.citations == ["[pid_a | PID A | page 1 | a-1]"]


def test_chat_still_rejects_unknown_pipe_separated_citation(monkeypatch):
    class UnknownCitationProvider:
        def complete(self, prompt):
            return LLMResponse(
                "Claim [pid_a | PID A | page 1 | a-1] [invented | citation]",
                1,
                1,
                0.0,
            )

    _stub_search(monkeypatch, [Excerpt("NOTE 24", "pid_a", "A", 1, "a-1")])
    result = GroundedChatService(UnknownCitationProvider()).answer("note?")

    assert result.status == "unsupported"
