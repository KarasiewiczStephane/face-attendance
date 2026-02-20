"""Structured logging setup for the face attendance system.

Provides a consistent logging format across all modules with
configurable log levels.
"""

import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with structured formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger
