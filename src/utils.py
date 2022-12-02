import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
