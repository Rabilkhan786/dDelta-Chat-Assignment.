"""Structured logging primitives shared by every pipeline stage."""

import logging
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
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
        for handler in (logging.StreamHandler(), logging.FileHandler(output_path, encoding="utf-8")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
    return RequestLoggerAdapter(logger, {"request_id": request_id or str(uuid.uuid4())})


def get_logger(name: str, request_id: str | None = None) -> RequestLoggerAdapter:
    """Compatibility entry point for module-scoped structured loggers."""
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
    """Log duration and failure context for a traceable pipeline stage."""
    started = time.perf_counter()
    logger.info("stage_started", extra={"stage": name})
    try:
        yield
    except Exception:
        logger.exception("stage_failed", extra={"stage": name, "duration_ms": round((time.perf_counter() - started) * 1000, 2)})
        raise
    logger.info("stage_completed", extra={"stage": name, "duration_ms": round((time.perf_counter() - started) * 1000, 2)})
