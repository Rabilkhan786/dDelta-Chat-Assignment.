"""Tokenize user questions for exact technical-identifier retrieval."""

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
