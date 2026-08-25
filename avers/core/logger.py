"""Logging configuration for AVERS."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "avers",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a logger for AVERS.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "avers") -> logging.Logger:
    """Get or create a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
