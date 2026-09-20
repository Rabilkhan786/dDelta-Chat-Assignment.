"""Canonical models, configuration, and structured logging tests."""

import json
import logging

import pytest
from pydantic import ValidationError

from src.canonical.model import CanonicalDocument, DocumentMetadata, Element, ElementType, Page
from src.canonical.serialization import write_canonical_document
from src.config.settings import PROJECT_ROOT, RetrievalConfig, project_path
from src.observability.logging import (
    RequestLoggerAdapter,
    RequestTraceHandler,
    get_logger,
    request_context,
    stage,
)

# Project


def _document() -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(
            document_id="doc-1",
            pid="PID-1",
            file_name="drawing.pdf",
            file_type="pdf",
            revision="A",
        ),
        pages=[
            Page(
                page_number=1,
                width=100,
                height=200,
                elements=[
                    Element(
                        id="p1_l1",
                        page_number=1,
                        type=ElementType.NOTE,
                        text="NOTE 1",
                    )
                ],
            )
        ],
    )


def test_canonical_document_writes_readable_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "document.json"

    write_canonical_document(_document(), destination)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["metadata"]["revision"] == "A"
    assert saved["pages"][0]["elements"][0]["type"] == "note"


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"page_number": 0},
        {"ocr_confidence": 1.1},
    ],
)
def test_element_rejects_invalid_canonical_values(invalid_values) -> None:
    values = {
        "id": "bad",
        "page_number": 1,
        "type": ElementType.TEXT,
        "text": "value",
        **invalid_values,
    }
    with pytest.raises(ValidationError):
        Element(**values)


# Config


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


# Logging


class _RecordCollector(logging.Handler):
    """Collects emitted log records so a test can inspect their structured fields."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _logger_with_collector(name: str) -> tuple[logging.LoggerAdapter, _RecordCollector]:
    logger = get_logger(name, request_id="test-request")
    collector = _RecordCollector()
    logger.logger.addHandler(collector)
    return logger, collector


def test_stage_logs_start_and_completion_with_duration() -> None:
    """Every pipeline stage must log its start, its end, and how long it took."""
    logger, collector = _logger_with_collector("tests.stage_logging")
    with stage(logger, "unit_test_stage"):
        pass
    messages = {record.message: record for record in collector.records}
    assert "stage_started" in messages
    assert "stage_completed" in messages
    assert messages["stage_completed"].stage == "unit_test_stage"
    assert messages["stage_completed"].duration_ms >= 0


def test_stage_logs_failure_and_still_reraises() -> None:
    """A failing stage must be visible in the logs, not swallowed."""
    logger, collector = _logger_with_collector("tests.stage_logging_failure")
    with pytest.raises(ValueError):
        with stage(logger, "failing_stage"):
            raise ValueError("boom")
    messages = {record.message: record for record in collector.records}
    assert "stage_failed" in messages


def test_request_context_binds_request_id_for_nested_logs() -> None:
    """A request ID set once must appear on every log emitted inside that context."""
    logger, collector = _logger_with_collector("tests.request_context")
    with request_context("req-xyz"):
        logger.info("nested_event")
    event = next(record for record in collector.records if record.message == "nested_event")
    assert event.request_id == "req-xyz"


def test_request_trace_handler_writes_one_request_file(tmp_path) -> None:
    logger = logging.getLogger("tests.request_trace_file")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RequestTraceHandler(tmp_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    adapter = RequestLoggerAdapter(logger, {"request_id": "fallback"})

    with request_context("req-123"):
        adapter.info("trace_event")

    trace_file = tmp_path / "req-123.jsonl"
    assert trace_file.exists()
    assert "trace_event" in trace_file.read_text(encoding="utf-8")
