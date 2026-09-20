"""Conservative query preparation shared by keyword search and routing.

Only spelling of known tag formats changes. Numbers, suffixes, negation, and
revision labels stay intact. No guessed synonyms or conversational memory.
"""

import re

# A small explicit allowlist avoids interpreting arbitrary words as tags.
TAG_PATTERN = re.compile(
    r"\b(PSV|PDI|PIT|PT|PI|FV|PV|XV|MOV|P|V|DN|NPS)[ -]*(\d+[A-Za-z]*)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.IGNORECASE)
DASHES = str.maketrans({character: "-" for character in "‐‑‒–—"})
STOP_WORDS = frozenset(
    "a an the what which is are was were do does did on of to for in and please".split()
)


def normalize_query(text: str) -> str:
    """Make PSV 9066A / PSV9066A searchable as PSV-9066A, preserving digits."""
    text = " ".join(text.translate(DASHES).split())

    def normalize_tag(match: re.Match) -> str:
        prefix, number = match.groups()
        separator = "" if prefix.upper() in {"DN", "NPS"} else "-"
        return f"{prefix.upper()}{separator}{number}"

    return TAG_PATTERN.sub(normalize_tag, text)


def search_queries(question: str, rewrite: bool = True) -> list[str]:
    """Always search the original; add at most one distinct normalized variant."""
    original = question.strip()
    if not original:
        raise ValueError("Question must not be empty.")
    normalized = normalize_query(original)
    if rewrite and normalized != original:
        return [original, normalized]
    return [original]


def keyword_tokens(text: str) -> list[str]:
    """Use identical normalization for queries and evidence, keeping tag suffixes.

    Include both a full hyphenated tag and its parts: some PDF labels occupy
    separate lines (PSV on one line and 9066A on another). Decimal values remain
    whole tokens, so 1.5 is not silently treated as 15 or separate 1 and 5.
    """
    tokens = []
    for token in TOKEN_PATTERN.findall(normalize_query(text).lower()):
        if token in STOP_WORDS:
            continue
        tokens.append(token)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens


def route_question(question: str) -> set[str] | None:
    """Prefer a source using whole words; ambiguous comparisons use all sources."""
    text = normalize_query(question).lower()
    revision_a = re.search(r"\b(?:(?:rev(?:ision)?|pid)\.? a|old revision|base revision)\b", text)
    revision_b = re.search(r"\b(?:(?:rev(?:ision)?|pid)\.? b|new revision|revised)\b", text)
    if (revision_a and revision_b) or re.search(
        r"\b(?:compare|comparison|differences?|between)\b", text
    ):
        return None
    if re.search(r"\b(?:changes?|changed|added|removed|modified|moved)\b", text):
        return {"delta_report"}
    if revision_a:
        return {"pid_a"}
    if revision_b:
        return {"pid_b"}
    return None
