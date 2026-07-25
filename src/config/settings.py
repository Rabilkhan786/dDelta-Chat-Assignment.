"""Centralised, validated configuration for Delta Chat."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str
    version: str


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)


class EmbeddingConfig(BaseModel):
    model: str
    device: str = "cpu"


class ChromaConfig(BaseModel):
    collection_name: str
    persist_directory: str


class RetrievalConfig(BaseModel):
    top_k: int = Field(gt=0)
    hybrid: bool = True
    use_bm25: bool = True
    use_semantic: bool = True


class ChunkingConfig(BaseModel):
    strategy: str


class PathsConfig(BaseModel):
    revision_a: str
    revision_b: str
    canonical_a: str
    canonical_b: str
    delta_json: str
    delta_markdown: str
    bm25_index: str
    bm25_documents: str


class AlignConfig(BaseModel):
    similarity_threshold: float = Field(ge=0.0, le=100.0)
    max_bbox_distance: float = Field(gt=0.0)


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    paths: PathsConfig
    align: AlignConfig


CONFIG_PATH = Path(__file__).with_name("config.yaml")
PROJECT_ROOT = CONFIG_PATH.parents[2]


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load and validate the checked-in, non-secret YAML configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return Settings.model_validate(yaml.safe_load(config_file))


def project_path(path: str | Path) -> Path:
    """Resolve a configured relative path from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


settings = load_settings()
