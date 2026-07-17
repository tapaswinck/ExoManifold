"""
Logging utilities for ExoManifold.

Provides a consistent logger configuration for the project.
"""

from __future__ import annotations

__all__ = [
    "get_logger"
]
import logging

from exomanifold.config import LOG_DIR

def get_logger(name: str)-> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name: str
        ususally __name__ of the calling module.


    Returns
    -------
    logging.Logger
        COnfigured logger.
    """

    logger = logging.getLogger(name)

    #Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    #Console output
    console_hanlder = logging.StreamHandler()
    console_hanlder.setFormatter(formatter)

    #File output
    file_handler = logging.FileHandler(
        LOG_DIR/ f"{name.replace('.', '_')}.log",
        encoding = "utf-8"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_hanlder)
    logger.addHandler(file_handler)

    return logger


