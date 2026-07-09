"""Loads instruments.yaml and params.yaml so the instrument universe and
tunable parameters never live as magic numbers scattered through the code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .data.exceptions import ConfigError

CONFIG_DIR = Path(__file__).parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping")
    return data


def load_instruments(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or CONFIG_DIR / "instruments.yaml")


def load_params(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or CONFIG_DIR / "params.yaml")
