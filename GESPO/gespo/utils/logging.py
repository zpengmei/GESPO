"""Logging configuration helpers."""
from __future__ import annotations

import logging
import os
from typing import Optional


def log_print(
    *args: object,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
    level: int = logging.INFO,
) -> None:
    """Log a print-style message using the configured logging handlers."""
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message = f"{message}{end}"
    message = message.rstrip("\n")
    logging.log(level, message)
    if flush:
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                continue


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
