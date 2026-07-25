import os
from pathlib import Path

list_of_files = [
    "README.md",
    ".env.example",
    ".gitignore",
    "Makefile",
    "docker-compose.yml",
    "pyproject.toml",

    # ---------------- src ---------------- #
    "src/ingest/__init__.py",
    "src/ingest/base.py",
    "src/ingest/pdf_native.py",
    "src/ingest/pdf_scanned.py",
    "src/ingest/dwg.py",

    "src/canonical/__init__.py",
    "src/canonical/model.py",

    "src/delta/__init__.py",
    "src/delta/align.py",
    "src/delta/engine.py",
    "src/delta/report.py",

    "src/chat/__init__.py",
    "src/chat/index.py",
    "src/chat/llm.py",
    "src/chat/answer.py",

    "src/markup/__init__.py",

    "src/observability/__init__.py",
    "src/observability/tracing.py",
    "src/observability/logging.py",

    # ---------------- evaluation ---------------- #
    "eval/datasets/.gitkeep",
    "eval/metrics.py",
    "eval/run_eval.py",

    # ---------------- sample data ---------------- #
    "data/samples/.gitkeep",

    # Optional working directories
    "data/input/.gitkeep",
    "data/output/.gitkeep",
    "data/reports/.gitkeep",
    "data/chroma_db/.gitkeep",

    # ---------------- tests ---------------- #
    "tests/__init__.py",

    # ---------------- docs ---------------- #
    "docs/.gitkeep",
]

for file_path in list_of_files:
    file_path = Path(file_path)

    if file_path.parent:
        os.makedirs(file_path.parent, exist_ok=True)

    if not file_path.exists():
        file_path.touch()