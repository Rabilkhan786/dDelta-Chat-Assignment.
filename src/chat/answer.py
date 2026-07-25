"""Grounded answer orchestration and citation enforcement."""

from dataclasses import dataclass

from src.chat.llm import ChatProvider, configured_provider
from src.chat.prompt import build_grounded_prompt, citation
from src.chat.retriever import HybridRetriever
from src.observability.logging import get_logger, stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class GroundedAnswer:
    """Chat output with the evidence citations exposed to callers."""

    text: str
    citations: list[str]


class GroundedChatService:
    """Retrieve evidence first, then invoke the one permitted LLM boundary."""

    def __init__(self, retriever: HybridRetriever | None = None, provider: ChatProvider | None = None) -> None:
        self.retriever = retriever or HybridRetriever()
        self.provider = provider

    def answer(self, question: str) -> GroundedAnswer:
        """Answer with evidence or explicitly state that retrieval found no support."""
        with stage(logger, "grounded_chat"):
            evidence = self.retriever.search(question)
            citations = [citation(item) for item in evidence]
            if not evidence:
                return GroundedAnswer("I cannot support an answer from the indexed PID A, PID B, or delta report.", [])
            prompt = build_grounded_prompt(question, evidence)
            provider = self.provider or configured_provider()
            response = provider.complete(prompt)
        return GroundedAnswer(response.text, citations)
