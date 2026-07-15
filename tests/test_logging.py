import pytest

from exomanifold.utils.logging import get_logger

@pytest.fixture
def logger():
    return get_logger("test_logger")

def test_logger_name(logger):
    assert logger.name == "test_logger"

def test_logger_has_handlers(logger):
    assert len(logger.handlers) == 2

def test_log_file_exists(logger):
    logger.info("Testing logger")