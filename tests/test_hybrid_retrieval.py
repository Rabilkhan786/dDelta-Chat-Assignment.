"""Unit tests for the deterministic pieces of hybrid retrieval."""

from types import SimpleNamespace

import pytest

from src.canonical.model import CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.chat import index
from src.chat.index import Excerpt, reciprocal_rank_fusion


def test_rrf_rewards_an_identifier_found_by_both_retrievers() -> None:
    scores = reciprocal_rank_fusion(
        [["PSV-9066", "P-101"], ["PSV-9066", "note"]],
        rrf_k=10,
    )
    assert scores["PSV-9066"] > scores["P-101"]


def test_keyword_tokens_match_compact_spaced_and_hyphenated_tags() -> None:
    compact = set(index.keyword_tokens("PSV9066A"))
    spaced = set(index.keyword_tokens("PSV 9066A"))
    hyphenated = set(index.keyword_tokens("PSV-9066A"))

    assert {"psv", "9066a"} <= compact & spaced & hyphenated
    assert "1.5" in index.keyword_tokens("1.5 bar")


def test_indexed_pid_text_has_no_retrieval_scaffolding() -> None:
    document = CanonicalDocument(
        metadata=DocumentMetadata(
            document_id="id",
            pid="pid-a",
            file_name="a.pdf",
            file_type="pdf",
            revision="A",
        ),
        pages=[
            Page(
                page_number=1,
                width=100,
                height=100,
                elements=[
                    Element(
                        id="a-1",
                        page_number=1,
                        type=ElementType.TEXT,
                        text="PSV-9066A",
                    )
                ],
            )
        ],
    )

    assert index._document_excerpts(document, "pid_a")[0].text == "PSV-9066A"


def test_keyword_search_ignores_revision_scaffolding_and_rewards_change_words() -> None:
    ordinary = Excerpt("pump pressure", "pid_b", "B", 1, "pump")
    existing_note = Excerpt("NOTE 33", "pid_b", "B", 1, "note-33")
    added_note = Excerpt(
        "Delta report change. added: Added note: NOTE 24 NEW BLOWDOWN VALVE",
        "delta_report",
        "B",
        1,
        "delta-3",
    )
    scores = index._keyword_scores(
        "What note was added in revision B?", [ordinary, existing_note, added_note]
    )

    assert index._excerpt_id(ordinary) not in scores
    assert scores[index._excerpt_id(added_note)] > scores[index._excerpt_id(existing_note)]


class FakeStore:
    """Exercise keyword search and fusion with controlled vector distances."""

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
            Excerpt(
                "Revision A document text: PSV-9066A pressure 1.5 bar",
                "pid_a",
                "A",
                1,
                "target",
            ),
            Excerpt(
                "Revision B document text: PSV-9066B pressure 15 bar",
                "pid_b",
                "B",
                1,
                "other",
            ),
        ]
    )
    monkeypatch.setattr(index, "_vector_store", lambda: store)
    monkeypatch.setattr(index.settings.reranker, "enabled", False)

    result = index.search("PSV 9066A 1.5 bar", top_k=1)

    assert result[0].element_id == "target"
    assert store.queries == ["PSV 9066A 1.5 bar"]


def test_no_keyword_overlap_and_weak_vectors_return_no_evidence(monkeypatch):
    store = FakeStore([Excerpt("Revision A document text: pump pressure", "pid_a", "A", 1, "pump")])
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


def test_delta_index_contains_summary_and_individual_changes() -> None:
    report = {
        "summary": {
            "actual_changes": 3,
            "modified": 1,
            "removed": 1,
            "added": 1,
            "moved": 0,
        },
        "entries": [
            {
                "delta_id": "delta-1",
                "change_type": "modified",
                "element_type": "text",
                "page_number": 1,
                "confidence": 0.8,
                "description": "text changed from '9066A' to '9066C'",
            },
            {
                "delta_id": "delta-2",
                "change_type": "removed",
                "element_type": "text",
                "page_number": 1,
                "confidence": 1.0,
                "description": "Removed text: 'MECHANICAL INTERLOCK'",
            },
            {
                "delta_id": "delta-3",
                "change_type": "added",
                "element_type": "note",
                "page_number": 1,
                "confidence": 1.0,
                "description": "Added note: 'NOTE 24'",
            },
        ],
    }

    excerpts = index._delta_excerpts(report, "revision_b", "B")

    assert [item.element_id for item in excerpts] == [
        "delta-summary",
        "delta-1",
        "delta-2",
        "delta-3",
    ]
    assert "3 changes detected" in excerpts[0].text
    assert "1 modified" in excerpts[0].text
    assert "1 removed" in excerpts[0].text
    assert "1 added" in excerpts[0].text
    assert "9066A" not in excerpts[0].text
    assert "MECHANICAL INTERLOCK" not in excerpts[0].text
    assert "NOTE 24" not in excerpts[0].text
    assert "9066A" in excerpts[1].text
    assert "MECHANICAL INTERLOCK" in excerpts[2].text
    assert "NOTE 24" in excerpts[3].text


def test_fusion_has_no_source_specific_boost() -> None:
    pid = Excerpt("same evidence", "pid_a", "A", 1, "pid")
    delta = Excerpt("same evidence", "delta_report", "B", 1, "delta-1")
    items = [pid, delta]

    lexical = {
        index._excerpt_id(pid): 2.0,
        index._excerpt_id(delta): 1.0,
    }
    semantic = {
        index._excerpt_id(pid): 1.0,
        index._excerpt_id(delta): 2.0,
    }

    results = index._fuse_candidates(items, lexical, semantic)

    assert results[0].score == results[1].score


def test_reranker_pool_reserves_candidates_from_each_retriever() -> None:
    lexical_target = Excerpt("exact identifier", "delta_report", "B", 1, "lexical")
    semantic_target = Excerpt("semantic meaning", "pid_b", "B", 1, "semantic")
    fused_target = Excerpt("supported by both", "pid_a", "A", 1, "fused")
    items = [lexical_target, semantic_target, fused_target]
    lexical = {
        index._excerpt_id(lexical_target): 3,
        index._excerpt_id(fused_target): 2,
    }
    semantic = {
        index._excerpt_id(semantic_target): 3,
        index._excerpt_id(fused_target): 2,
    }
    fused = index._fuse_candidates(items, lexical, semantic)

    pool = index._balanced_candidate_pool(items, fused, lexical, semantic, count=1)

    assert {item.element_id for item in pool} == {"lexical", "semantic", "fused"}


def test_candidate_pool_keeps_a_small_preferred_source_complete() -> None:
    summary = Excerpt("3 changes", "delta_report", "B", 1, "delta-summary")
    modified = Excerpt("modified", "delta_report", "B", 1, "delta-1")
    removed = Excerpt("removed", "delta_report", "B", 1, "delta-2")
    pid = Excerpt("other evidence", "pid_b", "B", 1, "pid")

    pool = index._balanced_candidate_pool(
        [summary, modified, removed, pid],
        [summary],
        {},
        {},
        count=3,
        preferred_source="delta_report",
    )

    assert {item.element_id for item in pool} == {
        "delta-summary",
        "delta-1",
        "delta-2",
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What changed in revision B?", "delta_report"),
        ("What does revision A say?", "pid_a"),
        ("Show revision B", "pid_b"),
        ("Compare revision A and revision B", None),
        ("Show the heat exchanger", None),
    ],
)
def test_query_routing_is_a_small_source_preference(query, expected) -> None:
    assert index.route_question(query) == expected


def test_source_preference_reorders_without_filtering() -> None:
    pid = Excerpt("PID evidence", "pid_b", "B", 1, "pid")
    delta = Excerpt("delta evidence", "delta_report", "B", 1, "delta")
    reordered = index.prefer_source([pid, delta], "delta_report")
    assert reordered == [delta, pid]


def test_delta_route_can_remove_pid_noise_without_emptying_results() -> None:
    pid = Excerpt("PID evidence", "pid_b", "B", 1, "pid")
    delta = Excerpt("delta evidence", "delta_report", "B", 1, "delta")

    assert index.prefer_source([pid, delta], "delta_report", only_preferred=True) == [delta]
    assert index.prefer_source([pid], "delta_report", only_preferred=True) == [pid]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What changed?", True),
        ("Summarize all changes in revision B", True),
        ("What changed on PSV-9066?", False),
        ("What callout was removed?", False),
        ("Compare revision A and revision B", False),
    ],
)
def test_broad_change_question_detection(query, expected) -> None:
    assert index.is_broad_change_question(query) is expected
