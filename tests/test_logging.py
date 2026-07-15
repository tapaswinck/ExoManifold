from exomanifold.utils.logging import get_logger

def test_logger():
    logger = get_logger("test")

    logger.info("Hello ExoManifold")
    
    assert logger.name == "test"

    