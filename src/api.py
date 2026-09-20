"""Optional single-process HTTP layer for the demo and local integrations."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.chat.answer import GroundedChatService
from src.config.settings import project_path, settings
from src.pipeline import DeltaPipeline, adapter_for

app = FastAPI(title="Delta Chat", version=settings.app.version)


class CompareRequest(BaseModel):
    """Two local PDF paths and an optional diagnostic adapter selection."""

    revision_a: str = str(project_path(settings.paths.revision_a))
    revision_b: str = str(project_path(settings.paths.revision_b))
    adapter: str = "auto"


class ChatRequest(BaseModel):
    """One grounded question over the most recently indexed document pair."""

    question: str


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the local API process is reachable."""
    return {"status": "ok", "service": settings.app.name}


@app.post("/compare")
def compare(request: CompareRequest) -> dict:
    """Ingest, compare, report, and index a document pair in one request."""
    try:
        result = DeltaPipeline(adapter_for(request.adapter)).run(Path(request.revision_a), Path(request.revision_b))
    except (FileNotFoundError, NotImplementedError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"request_id": result.request_id, "report": result.report, "indexed_excerpts": result.indexed_documents}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Answer a question only from retrieved evidence, with source citations."""
    try:
        answer = GroundedChatService().answer(request.question)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"request_id": answer.request_id, "answer": answer.text, "citations": answer.citations}


@app.get("/report")
def report() -> dict:
    """Return the latest machine-readable delta report without recomputing it."""
    report_path = project_path(settings.paths.delta_json)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No report exists. Run POST /compare first.")
    return json.loads(report_path.read_text(encoding="utf-8"))
