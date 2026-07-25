import logging
import uuid
from pathlib import Path

from pythonjsonlogger import jsonlogger


def setup_logger(
    name: str = "delta_chat",
    log_file: str = "logs/project.log",
    level: int = logging.INFO,
) -> logging.LoggerAdapter:
    """
    Configure and return a JSON structured logger with request_id support.
    """

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    request_id = str(uuid.uuid4())

    return logging.LoggerAdapter(
        logger,
        {"request_id": request_id},
    )