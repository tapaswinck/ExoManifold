import logging
from pathlib import Path

from exomanifold.config import LOG_DIR

def get_logger(name: str)->logging.Logger:
    """
    Create or return a configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logfile = logging.FileHandler(
        LOG_DIR / f"{name}.log", encoding = "utf-8"
    )

    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

    return logger

