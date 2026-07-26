"""Swappable LLM provider interface used only by grounded chat."""

from __future__ import annotations
from google import genai
from google.genai import types
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


class GeminiChatProvider:
    """AI implementation configured exclusively through config and environment."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Key is required for LLM-backed chat.")
        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt: str) -> LLMResponse:
        with stage(logger, "llm_completion"):
            response = self.client.models.generate_content(
                model=settings.llm.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.llm.temperature,
                    max_output_tokens=settings.llm.max_tokens,
                )
            )
        
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else None
        output_tokens = usage.candidates_token_count if usage else None
        total_tokens = usage.total_token_count if usage else None
        estimated_cost_usd = self._estimate_cost(input_tokens, output_tokens)
        
        logger.info("llm_completed", 
                    extra={
            "model": settings.llm.model,
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd
            }
                    )
        
        return LLMResponse(response.text or "", input_tokens, output_tokens, estimated_cost_usd)

    @staticmethod
    def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Estimate pay-as-you-go text-token cost from centralized model pricing."""
        if input_tokens is None or output_tokens is None:
            return None
        return round(
            input_tokens * settings.llm.input_cost_per_million_tokens_usd / 1_000_000
            + output_tokens * settings.llm.output_cost_per_million_tokens_usd / 1_000_000,
            8,
        )


def configured_provider() -> ChatProvider:
    """Resolve the configured provider without scattering provider selection."""
    if settings.llm.provider.lower() == "gemini":
        return GeminiChatProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")
