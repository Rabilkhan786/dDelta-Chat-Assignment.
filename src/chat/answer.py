"""Grounded answer orchestration and citation enforcement."""

import re
import unicodedata
import uuid
from dataclasses import dataclass

from src.chat import index
from src.chat.llm import ChatProvider, configured_provider
from src.chat.prompt import build_grounded_prompt, citation
from src.observability.logging import get_logger, request_context, stage

logger = get_logger(__name__)

CITATION_BRACKETS = str.maketrans(
    {
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
    }
)
BRACKET_PATTERN = re.compile(r"\[[^\[\]\n]+\]")
ELEMENT_ID_PATTERN = re.compile(r"(?:delta-(?:summary|\d+)|p\d+_l\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class GroundedAnswer:
    """Chat output with the evidence citations exposed to callers."""

    text: str
    citations: list[str]
    request_id: str
    status: str = "answered"


class GroundedChatService:
    """Retrieve evidence first, then invoke the one permitted LLM boundary."""

    def __init__(self, provider: ChatProvider | None = None) -> None:
        self.provider = provider

    def answer(self, question: str, request_id: str | None = None) -> GroundedAnswer:
        """Answer with evidence, or explicitly state that retrieval found no support."""
        request_id = request_id or str(uuid.uuid4())
        with request_context(request_id), stage(logger, "grounded_chat"):
            evidence = index.search(question)
            citations = [citation(item) for item in evidence]
            if not evidence:
                return GroundedAnswer(
                    "I cannot support an answer from the indexed PID A, PID B, or delta report.",
                    [],
                    request_id,
                    "unsupported",
                )

            prompt = build_grounded_prompt(question, evidence)
            try:
                provider = self.provider or configured_provider()
                response = provider.complete(prompt)
            except Exception:
                logger.exception(
                    "grounded_chat_provider_failed", extra={"evidence_count": len(evidence)}
                )
                return GroundedAnswer(
                    "I retrieved supporting evidence, but the configured LLM provider could not complete the request. "
                    "Check GROQ_API_KEY, billing, and provider availability before retrying.",
                    citations,
                    request_id,
                    "provider_error",
                )
            normalized_text, used, unknown = _normalize_citations(response.text, citations)
            if not used or unknown:
                logger.warning("answer_citations_rejected", extra={"citation_count": len(used)})
                return GroundedAnswer(
                    "I cannot support a cited answer from the retrieved evidence.",
                    [],
                    request_id,
                    "unsupported",
                )

            # Valid source references do not by themselves prove factual entailment.
            logger.info("answer_completed", extra={"citations_used": len(used)})
            return GroundedAnswer(normalized_text, used, request_id)


def _normalize_citations(text: str, allowed: list[str]) -> tuple[str, list[str], list[str]]:
    """Normalize citation typography while preserving exact source identity."""
    text = text.translate(CITATION_BRACKETS)
    allowed_by_key = {_citation_key(item): item for item in allowed}
    allowed_by_element = _unique_element_citations(allowed)
    used: list[str] = []
    unknown: list[str] = []

    def replace(match: re.Match) -> str:
        candidate = match.group(0)
        # Expand a provider's shortened element citation only when that ID is
        # unique in the retrieved evidence. Ordinary bracketed prose is kept.
        if "|" not in candidate:
            element_id = candidate[1:-1].strip()
            canonical = allowed_by_element.get(element_id.casefold())
            if canonical is not None:
                used.append(canonical)
                return canonical
            if ELEMENT_ID_PATTERN.fullmatch(element_id):
                unknown.append(candidate)
            return candidate
        canonical = allowed_by_key.get(_citation_key(candidate))
        if canonical is None:
            unknown.append(candidate)
            return candidate
        used.append(canonical)
        return canonical

    normalized_text = BRACKET_PATTERN.sub(replace, text)
    return normalized_text, list(dict.fromkeys(used)), unknown


def _citation_key(citation: str) -> str:
    """Treat Unicode spaces as formatting, never as a different source."""
    return " ".join(unicodedata.normalize("NFKC", citation).split())


def _unique_element_citations(allowed: list[str]) -> dict[str, str]:
    """Map unambiguous retrieved element IDs to their complete citations."""
    grouped: dict[str, list[str]] = {}
    for citation_text in allowed:
        parts = citation_text.strip("[]").split("|")
        if len(parts) < 4:
            continue
        element_id = parts[-1].strip().casefold()
        grouped.setdefault(element_id, []).append(citation_text)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}
