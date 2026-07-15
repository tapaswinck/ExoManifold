from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR/ "raw"
CACHE_DIR = DATA_DIR/ "cache"
PROCESSED_DIR = DATA_DIR/ "processed"
FOLDED_DATA_DIR = DATA_DIR/ "folded"
EMBEDDING_DIR = DATA_DIR/ "embeddings"

MODELS_DIR = PROJECT_ROOT/ "models"
RESULTS_DIR = PROJECT_ROOT/ "results"
FIGURES_DIR = PROJECT_ROOT/ "figures"
LOG_DIR= PROJECT_ROOT/"logs"


#Ensure directories exist

_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    CACHE_DIR,
    PROCESSED_DIR,
    FOLDED_DATA_DIR,
    EMBEDDING_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    LOG_DIR,
]

for directory in _DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

    