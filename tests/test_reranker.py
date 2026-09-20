"""Test reranking without downloading a model."""

from src.chat.index import Excerpt
from src.chat import rerank


class _FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.1 if "weak" in text else 0.9 for _, text in pairs]


def test_cross_encoder_reranker_changes_candidate_order(monkeypatch) -> None:
    monkeypatch.setattr(rerank, "_model", lambda: _FakeCrossEncoder())
    candidates = [Excerpt("weak evidence", "pid_a", "A", 1, "a"),
        Excerpt("strong evidence", "pid_b", "B", 1, "b")]
    ranked = rerank.rerank("question", candidates)
    assert ranked[0].element_id == "b"
