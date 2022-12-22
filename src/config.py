from os.path import dirname, realpath
from pathlib import Path

ROOT_DIR = Path(dirname(realpath(__file__))).parent

# Data directory
DATA_DIR = ROOT_DIR / "data"

# RAW data directory
RAW_DATA_DIR = DATA_DIR / "raw"

# PROCESSED data directory
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = ROOT_DIR / "src"

PLAYGROUND_DIR = SRC_DIR / "playground"
