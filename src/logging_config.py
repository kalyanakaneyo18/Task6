"""Shared logging configuration for API and scripts (Step 13)."""

import logging
from logging.handlers import RotatingFileHandler

from src.config import LOGS_DIR, API_LOG_PATH


def setup_logger(
    name: str = "houseprice",
    level: int = logging.INFO,
    log_file: str | None = API_LOG_PATH,
) -> logging.Logger:
    """Configure a logger that writes to console and a rotating file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
