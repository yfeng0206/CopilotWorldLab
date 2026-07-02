"""Minimal YAML config loader with attribute-style access.

Kept intentionally tiny: the pilot has few knobs and we prefer plain dicts over a
config framework. ``load_config`` returns a nested ``Config`` that supports both
``cfg.env.render_width`` and ``cfg["env"]["render_width"]`` access.
"""
from __future__ import annotations

import os
from typing import Any

import yaml


class Config(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        self[name] = value


def load_config(path: str) -> Config:
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)} in {path}")
    return Config(data)
