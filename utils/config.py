"""Read a YAML config from configs/."""

import yaml


def load(path):
    """Parse `path` and return the config as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)
