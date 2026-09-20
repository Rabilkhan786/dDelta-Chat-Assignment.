"""Tests for the swappable Groq provider boundary and telemetry."""

from types import SimpleNamespace

import pytest

from src.chat import llm
from src.chat.llm import GroqChatProvider
from src.config.settings import settings


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
