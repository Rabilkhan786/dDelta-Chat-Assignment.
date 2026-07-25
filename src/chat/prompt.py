"""Prompt construction isolated from retrieval and provider code."""

from src.chat.index import IndexedDocument


def citation(document: IndexedDocument) -> str:
    """Create a stable citation describing a PID/source and page location."""
    return f"[{document.source} | PID {document.pid} | page {document.page_number} | {document.element_id}]"


def build_grounded_prompt(question: str, context: list[IndexedDocument]) -> str:
    """Require a concise answer supported exclusively by supplied context."""
    evidence = "\n\n".join(f"{citation(item)}\n{item.text}" for item in context)
    return (
        "Answer only from the evidence below. If it does not support an answer, say so. "
        "Cite every factual statement using the supplied bracketed citation exactly.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence}"
    )
