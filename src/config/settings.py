from pathlib import Path
from typing import Any
import yaml
from box import ConfigBox
from pydantic import BaseModel


class AppConfig(BaseModel):
    name: str
    version: str


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int


class EmbeddingConfig(BaseModel):
    model: str
    device: str 


class ChromaConfig(BaseModel):
    collection_name: str
    persist_directory: str


class RetrievalConfig(BaseModel):
    top_k: int
    hybrid: bool
    use_bm25: bool
    use_semantic: bool


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


class Align(BaseModel):
    similarity_threshold: float 
    max_bbox_distance: float


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    paths: PathsConfig
    align: Align



# Load YAML Configuration

CONFIG_PATH = Path(__file__).parent / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    settings = ConfigBox(yaml.safe_load(file))