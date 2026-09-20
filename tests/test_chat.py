"""Grounded answers, prompts, reranking, and provider-boundary tests."""

from types import SimpleNamespace

import pytest

from src.chat import index, llm, rerank
from src.chat.answer import GroundedChatService
from src.chat.index import Excerpt
from src.chat.llm import GroqChatProvider, LLMResponse
from src.chat.prompt import build_grounded_prompt
from src.chat.query import keyword_tokens
from src.config.settings import settings

# Chat Answer


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


def test_chat_expands_unique_retrieved_element_citation(monkeypatch):
    class ShortCitationProvider:
        def complete(self, prompt):
            return LLMResponse("The callout was removed [delta-2]", 1, 1, 0.0)

    evidence = Excerpt("removed callout", "delta_report", "B", 1, "delta-2")
    _stub_search(monkeypatch, [evidence])
    result = GroundedChatService(ShortCitationProvider()).answer("What was removed?")

    assert result.status == "answered"
    assert result.text.endswith("[delta_report | PID B | page 1 | delta-2]")


def test_chat_rejects_unknown_element_citation(monkeypatch):
    class UnknownElementProvider:
        def complete(self, prompt):
            return LLMResponse(
                "Claim [delta_report | PID B | page 1 | delta-2] [delta-99]",
                1,
                1,
                0.0,
            )

    evidence = Excerpt("removed callout", "delta_report", "B", 1, "delta-2")
    _stub_search(monkeypatch, [evidence])
    result = GroundedChatService(UnknownElementProvider()).answer("What was removed?")

    assert result.status == "unsupported"


# Chat Support


def test_missing_groq_key_fails_with_a_clear_message(monkeypatch) -> None:
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqChatProvider()


def test_groq_completion_returns_text_tokens_and_cost() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content="grounded answer"))],
        )

    provider = object.__new__(GroqChatProvider)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = provider.complete("evidence prompt")

    assert response.text == "grounded answer"
    assert response.input_tokens == 100
    assert response.output_tokens == 20
    assert response.estimated_cost_usd is not None
    assert calls[0]["model"] == settings.llm.model


def test_cost_estimate_uses_configured_token_rates() -> None:
    assert GroqChatProvider._estimate_cost(1_000_000, 1_000_000) == 0.375
    assert GroqChatProvider._estimate_cost(None, 1) is None


def test_configured_provider_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setattr(settings.llm, "provider", "unknown")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm.configured_provider()


def test_configured_provider_builds_groq_provider(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(settings.llm, "provider", "groq")
    monkeypatch.setattr(llm, "GroqChatProvider", lambda: sentinel)
    assert llm.configured_provider() is sentinel


# Query tokenization


def test_keyword_tokens_match_common_identifier_formats() -> None:
    compact = set(keyword_tokens("PSV9066A"))
    spaced = set(keyword_tokens("PSV 9066A"))
    hyphenated = set(keyword_tokens("PSV-9066A"))

    assert {"psv", "9066a"} <= compact & spaced & hyphenated
    assert "1.5" in keyword_tokens("1.5 bar")


# Cross-encoder reranking


class _FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.1 if "weak" in text else 0.9 for _, text in pairs]


def test_cross_encoder_reranker_changes_candidate_order(monkeypatch) -> None:
    monkeypatch.setattr(rerank, "_model", lambda: _FakeCrossEncoder())
    candidates = [
        Excerpt("weak evidence", "pid_a", "A", 1, "a"),
        Excerpt("strong evidence", "pid_b", "B", 1, "b"),
    ]

    ranked = rerank.rerank("question", candidates)

    assert ranked[0].element_id == "b"
    assert len(ranked) == 2


def test_reranker_orders_candidates_without_dropping_negative_scores(monkeypatch):
    class ScoredModel:
        def predict(self, pairs):
            return [-5.0] + [2.0] * (len(pairs) - 1)

    monkeypatch.setattr(rerank, "_model", lambda: ScoredModel())
    candidates = [Excerpt(f"evidence {i}", "pid_a", "A", 1, str(i)) for i in range(8)]

    results = rerank.rerank("question", candidates)

    assert len(results) == 8
    assert results[-1].element_id == "0"


def test_reranker_returns_empty_input_without_loading_a_model(monkeypatch) -> None:
    monkeypatch.setattr(
        rerank,
        "_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )
    assert rerank.rerank("question", []) == []


# Grounded prompt construction


def test_prompt_preserves_question_and_limits_unnecessary_citations() -> None:
    question = "Could you walk me through 9066C in the newer drawing?"
    evidence = [Excerpt("9066C", "pid_b", "revision_b", 1, "p1_l873")]

    prompt = build_grounded_prompt(question, evidence)

    assert f"Question: {question}" in prompt
    assert "Use the minimum citations needed" in prompt
    assert "do not add change history unless the question asks for it" in prompt
    assert "[pid_b | PID revision_b | page 1 | p1_l873]" in prompt
