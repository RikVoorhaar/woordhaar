"""Logging configuration using loguru."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and route them to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure loguru logging with file handler and uvicorn interception.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files. Defaults to project root / logs
    """
    # Remove default handler
    logger.remove()

    # Determine log directory
    if log_dir is None:
        # Default to project root / logs
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # File handler with rotation
    logger.add(
        log_dir / "woordhaar.log",
        rotation="10 MB",
        retention=5,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe logging
    )

    # Console handler (stderr)
    logger.add(
        sys.stderr,
        level=log_level,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        colorize=True,
    )

    # Intercept uvicorn and other standard library logs
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set specific loggers to appropriate levels
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
