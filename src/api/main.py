"""HTTP surface for the Delta Chat pipeline: upload two PDF revisions (native or
scanned) to extract + diff + index them, then ask grounded questions over the result.

Runs the same `DeltaPipeline` and `GroundedChatService` the CLI (`main.py`) uses,
so the deterministic delta and citation-enforced chat behave identically either way.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chat.answer import GroundedChatService
from src.config.settings import project_path, settings
from src.observability.logging import get_logger
from src.pipeline import DeltaPipeline, adapter_for

logger = get_logger(__name__)

app = FastAPI(title="Delta Chat API", version=settings.app.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = project_path("data/input/uploads")


def _report_on_disk() -> dict | None:
    """Load the last generated delta report from disk, if one exists."""
    path = project_path(settings.paths.delta_json)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# In-memory cache of the most recent run, seeded from disk so state survives
# an API restart as long as a prior `run`/`/api/upload` wrote a report.
_state: dict[str, Any] = {"report": _report_on_disk(), "indexed_documents": None}
_state["ready"] = _state["report"] is not None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]


class UploadResponse(BaseModel):
    document_id: str
    summary: dict
    entries: list[dict]
    indexed_documents: int


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)


@app.get("/api/health")
def health() -> dict:
    """Report whether a document pair has been processed and chat is ready."""
    return {"status": "ok", "ready": bool(_state["ready"])}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(revision_a: UploadFile = File(...), revision_b: UploadFile = File(...)) -> UploadResponse:
    """Ingest two uploaded PDFs (native text or scanned/OCR), diff them, and index
    both revisions plus the delta for grounded chat."""
    for upload_file in (revision_a, revision_b):
        if Path(upload_file.filename or "").suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail=f"'{upload_file.filename}' is not a PDF file.")

    document_id = str(uuid.uuid4())
    request_dir = UPLOAD_DIR / document_id
    path_a = request_dir / (Path(revision_a.filename).name or "revision_a.pdf")
    path_b = request_dir / (Path(revision_b.filename).name or "revision_b.pdf")
    _save_upload(revision_a, path_a)
    _save_upload(revision_b, path_b)

    try:
        result = DeltaPipeline(adapter_for("auto"), request_id=document_id).run(path_a, path_b)
    except Exception as error:
        logger.exception("upload_pipeline_failed", extra={"document_id": document_id})
        raise HTTPException(status_code=422, detail=f"Could not process the uploaded PDFs: {error}") from error

    _state["report"] = result.report
    _state["indexed_documents"] = result.indexed_documents
    _state["ready"] = True

    return UploadResponse(
        document_id=document_id,
        summary=result.report["summary"],
        entries=result.report["entries"],
        indexed_documents=result.indexed_documents,
    )


@app.get("/api/report")
def report() -> dict:
    """Return the delta report from the most recent upload (or prior CLI `run`)."""
    if not _state["report"]:
        raise HTTPException(status_code=404, detail="No document pair has been processed yet.")
    return _state["report"]


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question grounded in the indexed revisions and delta report."""
    if not _state["ready"]:
        raise HTTPException(status_code=409, detail="Upload and process a document pair before asking questions.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        answer = GroundedChatService().answer(request.question)
    except RuntimeError as error:
        # e.g. GROQ_API_KEY missing/misconfigured -- a client-fixable setup error, not a server bug.
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ChatResponse(answer=answer.text, citations=answer.citations)
