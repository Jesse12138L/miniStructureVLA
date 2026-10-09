"""Read a YAML config from configs/."""

import yaml


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)
