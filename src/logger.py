"""
logger.py
---------
Central logging configuration for PulseAI.
Every script imports from here — nothing sets up its own logger.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger with the given name.

    Logs go to two places at once:
      - logs/YYYY-MM-DD.log   (one file per day, appended to)
      - console               (so you see output while running)

    Args:
        name: pass __name__ from the calling file.
              This makes the log show which file the message came from.

    Returns:
        A configured Python logger instance.
    """

    # Create logs/ folder if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # One log file per day — all scripts write to the same daily file
    # Example: logs/2026-06-28.log
    log_filename = datetime.now().strftime("%Y-%m-%d.log")
    log_filepath = os.path.join("logs", log_filename)

    # Create the logger
    logger = logging.getLogger(name)

    # Only add handlers once — avoids duplicate log lines if
    # get_logger() is called multiple times in the same run
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s  [%(levelname)s]  %(name)s  —  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Handler 1 — write to file
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)

        # Handler 2 — print to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger