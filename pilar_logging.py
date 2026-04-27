# -*- coding: utf-8 -*-
"""
PILAR -- Centralized logging configuration.
All modules should use: from pilar_logging import get_logger
"""
import logging
import os
import sys


def get_logger(name: str = "pilar") -> logging.Logger:
    """Return a named logger with PILAR's standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        _configure(logger)
    return logger


def _configure(logger: logging.Logger) -> None:
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+) — write to PILAR_LOG_DIR env var, or platform default
    try:
        _default_log_dir = (
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "PILAR")
            if os.name == "nt"
            else os.path.join(os.path.expanduser("~"), ".pilar", "logs")
        )
        log_dir = os.environ.get("PILAR_LOG_DIR", _default_log_dir)
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, "pilar.log"),
            encoding="utf-8",
            maxBytes=0,  # RotatingFileHandler would be better but keep it simple
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # file logging is best-effort

    logger.propagate = False
