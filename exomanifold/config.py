from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR/ "raw"
CACHE_DIR = DATA_DIR/ "cache"
PROCESSED_DIR = DATA_DIR/ "processed"
FOLDED_DIR = DATA_DIR/ "folded"

