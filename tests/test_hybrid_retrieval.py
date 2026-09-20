"""Unit tests for the deterministic pieces of hybrid retrieval."""

from types import SimpleNamespace

import pytest

from src.chat import index
from src.chat.index import Excerpt, reciprocal_rank_fusion, route_question


def test_rrf_rewards_an_identifier_found_by_both_retrievers() -> None:
    scores = reciprocal_rank_fusion(
        [["PSV-9066", "P-101"], ["PSV-9066", "note"]],
        rrf_k=10,
    )
    assert scores["PSV-9066"] > scores["P-101"]


def test_change_question_prefers_delta_report() -> None:
    assert route_question("What changed on PSV-9066?") == {"delta_report"}


def test_comparison_question_keeps_all_sources() -> None:
    assert route_question("Compare revision A and revision B") is None


def test_keyword_tokens_match_compact_spaced_and_hyphenated_tags() -> None:
    compact = set(index.keyword_tokens("PSV9066A"))
    spaced = set(index.keyword_tokens("PSV 9066A"))
    hyphenated = set(index.keyword_tokens("PSV-9066A"))
    assert {"psv", "9066a"} <= compact & spaced & hyphenated
    assert "1.5" in index.keyword_tokens("1.5 bar")


class FakeStore:
    """Exercise real keyword search and fusion with controlled vector distances."""

    def __init__(self, items, distance=100):
        self.items = items
        self.distance = distance
        self.queries = []

    def get(self, include):
        return {
            "documents": [item.text for item in self.items],
            "metadatas": [index._metadata(item) for item in self.items],
        }

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        return [
            (
                SimpleNamespace(
                    page_content=item.text,
                    metadata=index._metadata(item),
                ),
                self.distance,
            )
            for item in self.items[:k]
        ]


def test_exact_tag_survives_poor_vectors_and_small_corpus(monkeypatch):
    store = FakeStore(
        [
            Excerpt("PSV-9066A pressure 1.5 bar", "pid_a", "A", 1, "target"),
            Excerpt("PSV-9066B pressure 15 bar", "pid_b", "B", 1, "other"),
        ]
    )
    monkeypatch.setattr(index, "_vector_store", lambda: store)
    monkeypatch.setattr(index.settings.reranker, "enabled", False)

    result = index.search("PSV 9066A 1.5 bar", top_k=1)

    assert result[0].element_id == "target"
    assert store.queries == ["PSV 9066A 1.5 bar"]


def test_no_keyword_overlap_and_weak_vectors_return_no_evidence(monkeypatch):
    store = FakeStore([Excerpt("pump pressure", "pid_a", "A", 1, "pump")])
    monkeypatch.setattr(index, "_vector_store", lambda: store)
    monkeypatch.setattr(index.settings.reranker, "enabled", False)

    assert index.search("What is the lunar weather?") == []


def test_empty_index_and_invalid_limits(monkeypatch):
    monkeypatch.setattr(index, "_vector_store", lambda: FakeStore([]))

    assert index.search("pressure") == []
    with pytest.raises(ValueError, match="top_k"):
        index.search("pressure", top_k=0)
    with pytest.raises(ValueError, match="Query"):
        index.search("   ")


def test_delta_index_reads_generated_report_entries() -> None:
    report = {
        "entries": [
            {
                "delta_id": "delta-1",
                "change_type": "added",
                "element_type": "note",
                "page_number": 1,
                "confidence": 1.0,
                "description": "Added note: 'new'",
            }
        ]
    }

    excerpts = index._delta_excerpts(report, "B", "B")

    assert len(excerpts) == 1
    assert excerpts[0].element_id == "delta-1"
    assert excerpts[0].change_type == "added"
