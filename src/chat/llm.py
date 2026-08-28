"""Swappable LLM provider interface used only by grounded chat."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from groq import Groq

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


class GroqChatProvider:
    """Groq implementation, configured exclusively through config and environment."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for LLM-backed chat.")
        self.client = Groq(api_key=api_key)

    def complete(self, prompt: str) -> LLMResponse:
        with stage(logger, "llm_completion"):
            response = self.client.chat.completions.create(
                model=settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        estimated_cost_usd = self._estimate_cost(input_tokens, output_tokens)

        logger.info("llm_completed", extra={"model": settings.llm.model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "estimated_cost_usd": estimated_cost_usd})

        return LLMResponse(response.choices[0].message.content or "", input_tokens, output_tokens, estimated_cost_usd)

    @staticmethod
    def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Estimate token cost from the centralized model pricing in config.yaml."""
        if input_tokens is None or output_tokens is None:
            return None
        return round(
            input_tokens * settings.llm.input_cost_per_million_tokens_usd / 1_000_000
            + output_tokens * settings.llm.output_cost_per_million_tokens_usd / 1_000_000,
            8,
        )


def configured_provider() -> ChatProvider:
    """Resolve the configured provider without scattering provider selection."""
    if settings.llm.provider.lower() == "groq":
        return GroqChatProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")
