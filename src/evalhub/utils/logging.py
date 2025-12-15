"""Logging utilities for the EvalHub SDK."""

import logging
import sys
from typing import Any


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
    stream: Any = None,
) -> logging.Logger:
    """Set up logging configuration for the EvalHub SDK.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        stream: Output stream for logging (defaults to stdout)

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if stream is None:
        stream = sys.stdout

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=stream,
        force=True,
    )

    # Return logger for evalhub package
    logger = logging.getLogger("evalhub")
    logger.setLevel(getattr(logging, level.upper()))

    return logger
