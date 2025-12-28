"""Logging configuration helpers."""
from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(log_level: str = "INFO", log_file: str = "") -> None:
    """Configure package-wide logging format and handlers."""
    level = logging.INFO
    if log_level:
        level = logging.getLevelName(log_level.upper())
        if isinstance(level, str):
            level = logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
