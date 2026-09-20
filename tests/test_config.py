"""Tests for central configuration validation and path handling."""

import pytest
from pydantic import ValidationError

from src.config.settings import PROJECT_ROOT, RetrievalConfig, project_path


def test_project_path_resolves_relative_paths_from_repository() -> None:
    assert project_path("data/example.json") == PROJECT_ROOT / "data/example.json"


def test_project_path_preserves_absolute_paths(tmp_path) -> None:
    assert project_path(tmp_path) == tmp_path


def test_retrieval_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(
            top_k=0,
            candidate_k=10,
            rrf_k=60,
            minimum_vector_similarity=0.35,
        )
