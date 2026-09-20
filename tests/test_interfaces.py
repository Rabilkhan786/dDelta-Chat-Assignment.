"""API, CLI, pipeline, and PDF markup integration tests."""

import json
from types import SimpleNamespace

import pymupdf
from fastapi.testclient import TestClient

import main
from src import api
from src.api import app
from src.canonical.model import BoundingBox, DeltaEntry, DeltaType, ElementType
from src.chat.answer import GroundedAnswer
from src.config.settings import settings
from src.markup.pdf import write_markup
from src.pipeline import DeltaPipeline, adapter_for

# Api

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


# Cli


class _ConsoleStream:
    def __init__(self):
        self.settings = None

    def reconfigure(self, **settings):
        self.settings = settings


def test_run_can_chat_with_the_pipeline_request_id(monkeypatch):
    result = SimpleNamespace(
        report={"summary": {"actual_changes": 3}}, indexed_documents=8, request_id="demo-request"
    )
    monkeypatch.setattr(main.DeltaPipeline, "run", lambda *args: result)
    calls = []

    def answer(self, question, request_id=None):
        calls.append((question, request_id))
        return GroundedAnswer("No supporting evidence", [], request_id, "unsupported")

    monkeypatch.setattr(main.GroundedChatService, "answer", answer)
    arguments = main.parser().parse_args(["run", "--question", "What changed?"])
    arguments.func(arguments)
    assert calls == [("What changed?", "demo-request")]


def test_plain_run_does_not_require_a_question():
    assert main.parser().parse_args(["run"]).question is None


def test_chat_command_prints_the_answer(monkeypatch, capsys):
    answer = GroundedAnswer("Grounded answer", ["[source]"], "request")
    monkeypatch.setattr(main.GroundedChatService, "answer", lambda self, question: answer)

    arguments = main.parser().parse_args(["chat", "question"])
    arguments.func(arguments)

    output = capsys.readouterr().out
    assert "Grounded answer" in output
    assert "[source]" in output


def test_console_uses_utf8_for_model_punctuation(monkeypatch):
    output = _ConsoleStream()
    errors = _ConsoleStream()
    monkeypatch.setattr(main.sys, "stdout", output)
    monkeypatch.setattr(main.sys, "stderr", errors)
    main.configure_console_output()
    assert output.settings == {"encoding": "utf-8", "errors": "replace"}
    assert errors.settings == {"encoding": "utf-8", "errors": "replace"}


# Pipeline


def test_scanned_and_native_revisions_use_the_same_delta_pipeline(tmp_path, monkeypatch):
    native_a = tmp_path / "native_a.pdf"
    native_b = tmp_path / "native_b.pdf"
    scanned_a = tmp_path / "scanned_a.pdf"
    for destination, pressure in [(native_a, 10), (native_b, 12)]:
        with pymupdf.open() as document:
            page = document.new_page(width=400, height=300)
            for y, text in [
                (50, "PSV-9066A"),
                (100, f"Pressure {pressure} bar"),
                (150, "NOTE 1: INSPECT VALVE"),
            ]:
                page.insert_text((40, y), text, fontsize=11)
            document.save(destination)
    with pymupdf.open(native_a) as native, pymupdf.open() as scan:
        page = scan.new_page(width=400, height=300)
        page.insert_image(page.rect, pixmap=native[0].get_pixmap(matrix=pymupdf.Matrix(3, 3)))
        scan.save(scanned_a)

    for field, filename in [
        ("canonical_a", "a.json"),
        ("canonical_b", "b.json"),
        ("delta_json", "custom_delta.json"),
        ("delta_markdown", "custom_delta.md"),
        ("delta_markup", "markup.pdf"),
    ]:
        monkeypatch.setattr(settings.paths, field, str(tmp_path / filename))
    # Model-backed indexing has a separate real-model smoke check. This test
    # verifies ingestion through report/markup and the canonical indexing seam.
    indexed = []

    def capture_index(old, new, report):
        indexed.append((old, new, report))
        return len(report["entries"])

    monkeypatch.setattr("src.pipeline.build_index", capture_index)
    result = DeltaPipeline(adapter_for("auto")).run(scanned_a, native_b)
    assert all(e.source == "ocr" for e in result.pid_a.pages[0].elements)
    changes = [entry for entry in result.deltas if entry.change_type.value != "unchanged"]
    assert len(changes) == 1
    assert changes[0].change_type.value == "modified"
    assert "10 bar" in changes[0].description and "12 bar" in changes[0].description
    assert result.report["revision_compatibility"]["compatible"]
    assert len(result.report["entries"]) == 1
    assert result.report["entries"][0]["delta_id"] == "delta-1"
    assert result.report["entries"][0]["location_revision"] == "B"
    assert indexed[0][0] == result.pid_a
    assert (tmp_path / "custom_delta.json").exists()
    assert (tmp_path / "custom_delta.md").exists()
    assert result.markup_path.exists()


# Markup


def test_markup_adds_one_annotation_for_a_changed_bbox(tmp_path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "markup.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=200)
        document.save(source)
    delta = DeltaEntry(
        change_type=DeltaType.ADDED,
        element_type=ElementType.TEXT,
        page_number=1,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Added text: 'new note'",
        confidence=1,
    )
    write_markup(source, destination, [delta])
    with pymupdf.open(destination) as marked:
        assert marked[0].first_annot is not None


def test_markup_skips_a_removed_page_not_present_in_revision_b(tmp_path):
    source = tmp_path / "revision_b.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(source)
    delta = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=2,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Removed page content",
        confidence=1,
    )
    destination = write_markup(source, tmp_path / "marked.pdf", [delta])
    with pymupdf.open(destination) as marked:
        assert len(marked) == 1
        assert marked[0].first_annot is None


def test_markup_does_not_draw_removed_content_on_revision_b(tmp_path):
    source = tmp_path / "revision_b.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=200)
        document.save(source)

    removed = DeltaEntry(
        change_type=DeltaType.REMOVED,
        element_type=ElementType.TEXT,
        page_number=1,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Removed text: 'old note'",
        confidence=1,
    )

    destination = write_markup(source, tmp_path / "marked.pdf", [removed])

    with pymupdf.open(destination) as marked:
        assert marked[0].first_annot is None


def test_markup_skips_changed_content_outside_revision_b_pages(tmp_path) -> None:
    source = tmp_path / "revision_b.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=200)
        document.save(source)
    added = DeltaEntry(
        change_type=DeltaType.ADDED,
        element_type=ElementType.TEXT,
        page_number=2,
        region=BoundingBox(x0=10, y0=10, x1=50, y1=25),
        description="Added text",
        confidence=1,
    )

    destination = write_markup(source, tmp_path / "marked.pdf", [added])

    with pymupdf.open(destination) as marked:
        assert marked[0].first_annot is None
