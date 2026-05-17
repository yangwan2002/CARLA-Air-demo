"""
logging_utils.py
----------------
Tiny wrapper around Python's logging module so every module shares the same
console + file handler configuration.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Optional[str | os.PathLike] = None,
) -> logging.Logger:
    """Configure the root logger exactly once.

    Returns the root logger so callers can immediately log from `main`.
    """
    root = logging.getLogger()
    # If `setup_logging` is called twice, just update level + return.
    if getattr(root, "_air_ground_relay_configured", False):
        root.setLevel(level)
        return root

    root.setLevel(level)

    # Clear any pre-existing handlers (e.g. when run inside a Jupyter notebook).
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root._air_ground_relay_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
