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
    assert len(ranked) == 2


def test_reranker_orders_candidates_without_dropping_negative_scores(monkeypatch):
    class ScoredModel:
        def predict(self, pairs):
            return [-5.0] + [2.0] * (len(pairs) - 1)

    monkeypatch.setattr(rerank, "_model", lambda: ScoredModel())
    candidates = [Excerpt(f"evidence {i}", "pid_a", "A", 1, str(i)) for i in range(8)]

    results = rerank.rerank("question", candidates)

    assert len(results) == 8
    assert results[-1].element_id == "0"


def test_reranker_returns_empty_input_without_loading_a_model(monkeypatch) -> None:
    monkeypatch.setattr(
        rerank,
        "_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )
    assert rerank.rerank("question", []) == []
