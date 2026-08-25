"""Configuration loading and paper-aligned defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

BACKBONE_ALIASES = {
    "resnet": "resnet101",
    "resnet101": "resnet101",
    "densenet": "densenet121",
    "densenet121": "densenet121",
    "vit": "vit_base",
    "vit_base": "vit_base",
    "swin": "swin_base",
    "swin_base": "swin_base",
}


def repository_root() -> Path:
    """Return the repository root from an editable or source installation."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    source_checkout = repository_root() / "configs" / "paper.yaml"
    if source_checkout.is_file():
        return source_checkout
    packaged = Path(__file__).with_name("paper.yaml")
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("The packaged paper.yaml configuration is missing.")


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the paper configuration and optionally overlay a user YAML file."""
    default_path = default_config_path()
    with default_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    if path is None:
        return config

    user_path = Path(path).expanduser().resolve()
    with user_path.open("r", encoding="utf-8") as stream:
        user_config = yaml.safe_load(stream) or {}
    return _deep_merge(config, user_config)


def normalize_backbone_name(name: str) -> str:
    """Return the canonical paper backbone name."""
    key = name.strip().lower().replace("-", "_")
    if key not in BACKBONE_ALIASES:
        choices = ", ".join(sorted(set(BACKBONE_ALIASES.values())))
        raise ValueError(f"Unsupported backbone '{name}'. Choose one of: {choices}")
    return BACKBONE_ALIASES[key]
