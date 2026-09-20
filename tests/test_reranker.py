"""Test reranking without downloading a model."""

from src.chat import rerank
from src.chat.index import Excerpt


class _FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.1 if "weak" in text else 0.9 for _, text in pairs]


def test_cross_encoder_reranker_changes_candidate_order(monkeypatch) -> None:
    monkeypatch.setattr(rerank, "_model", lambda: _FakeCrossEncoder())
    candidates = [
        Excerpt("weak evidence", "pid_a", "A", 1, "a"),
        Excerpt("strong evidence", "pid_b", "B", 1, "b"),
    ]
    ranked = rerank.rerank("question", candidates)
    assert ranked[0].element_id == "b"


def test_reranker_filters_weak_evidence_without_a_second_top_k_limit(monkeypatch):
    class ScoredModel:
        def predict(self, pairs):
            return [-5.0] + [2.0] * (len(pairs) - 1)

    monkeypatch.setattr(rerank, "_model", lambda: ScoredModel())
    candidates = [Excerpt(f"evidence {i}", "pid_a", "A", 1, str(i)) for i in range(8)]
    results = rerank.rerank("question", candidates)
    assert len(results) == 7
    assert "0" not in [item.element_id for item in results]
