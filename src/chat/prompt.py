"""Prompt construction isolated from retrieval and provider code."""

from src.chat.index import IndexedDocument


def citation(document: IndexedDocument) -> str:
    """Create a stable citation describing a PID/source and page location."""
    return f"[{document.source} | PID {document.pid} | page {document.page_number} | {document.element_id}]"


def build_grounded_prompt(question: str, context: list[IndexedDocument]) -> str:
    
    evidence = "\n\n".join(
        f"{citation(item)}\n{item.text}" for item in context
    )

    return (
        "You are an engineering document assistant.\n\n"
        "Answer ONLY from the supplied evidence.\n"
        "Do NOT use external knowledge.\n"
        "Do NOT guess or infer engineering intent.\n"
        "Treat delta entries literally.\n"
        "An ADDED text entry does NOT necessarily mean an equipment, pipeline, or location was added.\n"
        "A MODIFIED text entry does NOT necessarily indicate an engineering change beyond the text shown.\n"
        "If the evidence is insufficient, clearly state that.\n"
        "Cite every factual statement using the supplied bracketed citations exactly.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence}"
    )
