"""Tests for the optional single-process HTTP interface."""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src import api
from src.api import app

client = TestClient(app)


def test_health_endpoint_is_available() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_compare_returns_report_and_index_count(monkeypatch) -> None:
    class FakePipeline:
        def __init__(self, adapter):
            self.adapter = adapter

        def run(self, revision_a, revision_b):
            return SimpleNamespace(
                request_id="request-1",
                report={"summary": {"actual_changes": 1}},
                indexed_documents=12,
            )

    monkeypatch.setattr(api, "adapter_for", lambda name: "adapter")
    monkeypatch.setattr(api, "DeltaPipeline", FakePipeline)

    response = client.post(
        "/compare",
        json={"revision_a": "a.pdf", "revision_b": "b.pdf", "adapter": "auto"},
    )

    assert response.status_code == 200
    assert response.json()["indexed_excerpts"] == 12


def test_compare_returns_bad_request_for_invalid_adapter() -> None:
    response = client.post(
        "/compare",
        json={"revision_a": "a.pdf", "revision_b": "b.pdf", "adapter": "invalid"},
    )
    assert response.status_code == 400


def test_chat_returns_grounded_answer(monkeypatch) -> None:
    answer = SimpleNamespace(
        request_id="request-2",
        text="answer",
        citations=["[pid_a | PID A | page 1 | p1_l1]"],
        status="answered",
    )
    monkeypatch.setattr(api.GroundedChatService, "answer", lambda self, question: answer)

    response = client.post("/chat", json={"question": "What is shown?"})

    assert response.status_code == 200
    assert response.json()["status"] == "answered"


def test_chat_returns_bad_request_for_invalid_question(monkeypatch) -> None:
    def fail(self, question):
        raise ValueError("Query must not be empty.")

    monkeypatch.setattr(api.GroundedChatService, "answer", fail)
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 400


def test_report_returns_latest_json(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"summary": {"actual_changes": 2}}), encoding="utf-8")
    monkeypatch.setattr(api.settings.paths, "delta_json", str(report_path))

    response = client.get("/report")

    assert response.status_code == 200
    assert response.json()["summary"]["actual_changes"] == 2


def test_report_returns_not_found_when_compare_has_not_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api.settings.paths, "delta_json", str(tmp_path / "missing.json"))
    response = client.get("/report")
    assert response.status_code == 404
