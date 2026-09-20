import logging

import pytest

from src.observability.logging import RequestLoggerAdapter, RequestTraceHandler, get_logger, request_context, stage


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
