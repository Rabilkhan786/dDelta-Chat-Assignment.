"""Tests for the small grounded-answer prompt."""

from src.chat.index import Excerpt
from src.chat.prompt import build_grounded_prompt


def test_prompt_preserves_question_and_limits_unnecessary_citations() -> None:
    question = "Could you walk me through 9066C in the newer drawing?"
    evidence = [Excerpt("9066C", "pid_b", "revision_b", 1, "p1_l873")]

    prompt = build_grounded_prompt(question, evidence)

    assert f"Question: {question}" in prompt
    assert "Use the minimum citations needed" in prompt
    assert "do not add change history unless the question asks for it" in prompt
    assert "[pid_b | PID revision_b | page 1 | p1_l873]" in prompt
