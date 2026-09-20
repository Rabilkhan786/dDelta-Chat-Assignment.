"""Prompt construction isolated from retrieval and provider code."""

from src.chat.index import Excerpt


def citation(excerpt: Excerpt) -> str:
    """Create a stable citation describing a PID/source and page location."""
    return f"[{excerpt.source} | PID {excerpt.pid} | page {excerpt.page_number} | {excerpt.element_id}]"


def build_grounded_prompt(question: str, evidence: list[Excerpt]) -> str:
    """Require a concise answer supported exclusively by the retrieved evidence."""
    context = "\n\n".join(f"{citation(item)}\n{item.text}" for item in evidence)
    return (
        "You are an engineering document assistant. Answer only from the evidence below; "
        "do not use outside knowledge or guess at engineering intent. PID excerpts describe "
        "document content, while delta-report excerpts describe detected revision changes. "
        "Do not treat ordinary PID text as proof that something changed. Treat delta entries "
        "literally: an added or modified text entry does not by itself mean equipment, "
        "piping, or a location changed. If the evidence does not support an answer, say so. "
        "Cite every factual statement using the supplied bracketed citation exactly.\n\n"
        f"Question: {question}\n\nEvidence:\n{context}"
    )
