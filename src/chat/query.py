"""Small deterministic query helpers used by hybrid retrieval."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z]+\d+[A-Za-z]*|\d+(?:\.\d+)?[A-Za-z]*|[A-Za-z]+")
STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "did",
        "do",
        "document",
        "does",
        "for",
        "in",
        "is",
        "of",
        "on",
        "please",
        "revision",
        "say",
        "says",
        "text",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
    }
)
GENERIC_CHANGE_WORDS = {
    "all",
    "change",
    "changed",
    "changes",
    "difference",
    "differences",
    "modified",
    "new",
    "removed",
    "summarize",
    "summary",
}


def keyword_tokens(text: str) -> list[str]:
    """Tokenize technical identifiers without rewriting the question."""
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        if token in STOP_WORDS or len(token) == 1:
            continue

        tokens.append(token)
        compact = re.fullmatch(r"([a-z]+)(\d+(?:\.\d+)?[a-z]*)", token)
        if compact:
            tokens.extend(compact.groups())

    return list(dict.fromkeys(tokens))


def route_question(query: str) -> str | None:
    """Choose a simple evidence preference from explicit words in the question."""
    text = query.lower()
    revision_a = re.search(r"\b(?:(?:rev(?:ision)?|pid)\.? a|old revision|base revision)\b", text)
    revision_b = re.search(r"\b(?:(?:rev(?:ision)?|pid)\.? b|new revision|revised)\b", text)
    if (revision_a and revision_b) or re.search(
        r"\b(?:compare|comparison|differences?|between)\b", text
    ):
        return None
    if re.search(r"\b(?:changes?|changed|added|removed|modified|moved)\b", text):
        return "delta_report"
    if revision_a:
        return "pid_a"
    if revision_b:
        return "pid_b"
    return None


def is_broad_change_question(query: str) -> bool:
    """Return true when a change question has no specific technical subject."""
    if route_question(query) != "delta_report":
        return False
    return not (set(keyword_tokens(query)) - GENERIC_CHANGE_WORDS)
