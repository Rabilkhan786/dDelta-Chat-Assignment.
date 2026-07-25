"""Swappable LLM provider interface used only by grounded chat."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from src.config.settings import settings
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral response and telemetry needed by observability."""

    text: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None


class ChatProvider(Protocol):
    """A provider boundary that keeps SDK calls out of chat orchestration."""

    def complete(self, prompt: str) -> LLMResponse: ...


class OpenAIChatProvider:
    """OpenAI implementation configured exclusively through config and environment."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM-backed chat.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def complete(self, prompt: str) -> LLMResponse:
        with stage(logger, "llm_completion"):
            response = self.client.chat.completions.create(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        usage = response.usage
        logger.info("llm_completed", extra={"model": settings.llm.model,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None, "estimated_cost_usd": None})
        return LLMResponse(response.choices[0].message.content or "", usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None, None)


def configured_provider() -> ChatProvider:
    """Resolve the configured provider without scattering provider selection."""
    if settings.llm.provider.lower() == "openai":
        return OpenAIChatProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")
