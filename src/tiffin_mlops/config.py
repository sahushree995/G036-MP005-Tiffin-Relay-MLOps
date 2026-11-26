"""Load project settings from params.yaml."""
from pathlib import Path

import yaml

PARAMS_PATH = Path(__file__).resolve().parents[2] / "params.yaml"


def load_params(path=PARAMS_PATH):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
