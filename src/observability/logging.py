"""Structured logging primitives shared by every pipeline stage."""

import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from pythonjsonlogger.json import JsonFormatter

from src.config.settings import PROJECT_ROOT

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Resolve the active request ID at emit time for every pipeline module."""

    def process(self, message: str, kwargs: dict) -> tuple[str, dict]:
        extra = dict(kwargs.get("extra", {}))
        extra.setdefault("request_id", _request_id.get() or self.extra["request_id"])
        kwargs["extra"] = extra
        return message, kwargs


class RequestTraceHandler(logging.Handler):
    """Write each structured event to a request-specific JSONL trace file."""

    def __init__(self, trace_dir: Path) -> None:
        super().__init__()
        self.trace_dir = trace_dir

    def emit(self, record: logging.LogRecord) -> None:
        request_id = getattr(record, "request_id", None)
        if not request_id:
            return

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(request_id))
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.trace_dir / f"{safe_id}.jsonl"
            with trace_path.open("a", encoding="utf-8", newline="\n") as trace_file:
                trace_file.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str = "delta_chat",
    log_file: str = "logs/project.log",
    level: int = logging.INFO,
    request_id: str | None = None,
) -> RequestLoggerAdapter:
    """Return a JSON logger carrying a request identifier."""
    output_path = PROJECT_ROOT / log_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )
        handlers = (
            logging.StreamHandler(),
            logging.FileHandler(output_path, encoding="utf-8"),
            RequestTraceHandler(PROJECT_ROOT / "logs" / "traces"),
        )
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False

    return RequestLoggerAdapter(
        logger,
        {"request_id": request_id or str(uuid.uuid4())},
    )


def get_logger(name: str, request_id: str | None = None) -> RequestLoggerAdapter:
    """Return a structured logger for one module."""
    return setup_logger(name=name, request_id=request_id)


@contextmanager
def request_context(request_id: str) -> Iterator[None]:
    """Bind one correlation ID across all nested module logs in a request."""
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


@contextmanager
def stage(logger: logging.LoggerAdapter, name: str) -> Iterator[None]:
    """Log stage start, completion/failure, and duration."""
    started = time.perf_counter()
    logger.info("stage_started", extra={"stage": name})
    try:
        yield
    except Exception:
        logger.exception(
            "stage_failed",
            extra={
                "stage": name,
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
            },
        )
        raise

    logger.info(
        "stage_completed",
        extra={
            "stage": name,
            "duration_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
        },
    )
