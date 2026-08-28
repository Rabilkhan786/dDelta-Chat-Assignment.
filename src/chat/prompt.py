"""Prompt construction isolated from retrieval and provider code."""

from src.chat.index import Excerpt


def citation(excerpt: Excerpt) -> str:
    """Create a stable citation describing a PID/source and page location."""
    return f"[{excerpt.source} | PID {excerpt.pid} | page {excerpt.page_number} | {excerpt.element_id}]"


def build_grounded_prompt(question: str, evidence: list[Excerpt]) -> str:
    """Require a concise answer supported exclusively by the retrieved evidence."""
    context = "\n\n".join(f"{citation(item)}\n{item.text}" for item in evidence)
    return (
        "Answer only from the evidence below. If it does not support an answer, say so. "
        "Cite every factual statement using the supplied bracketed citation exactly.\n\n"
        f"Question: {question}\n\nEvidence:\n{context}"
    )
