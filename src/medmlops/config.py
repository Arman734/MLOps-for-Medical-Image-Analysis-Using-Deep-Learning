"""Configuration loading with dot-access dictionaries and dotenv support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("MEDMLOPS_ROOT") or Path(__file__).resolve().parents[2]).resolve()


def _dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Env:
    """Tiny .env reader that does not override real environment variables."""

    def __init__(self, path: Path | None = None) -> None:
        self._values = _dotenv_values(path or PROJECT_ROOT / ".env")

    def get(self, key: str, default: str | None = None) -> str | None:
        return os.environ.get(key) or self._values.get(key) or default

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config(dict):
    """Nested dict with attribute access, e.g. cfg.data.image_size."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def get_path(self, key: str, default: Any = None) -> Any:
        """Fetch a dotted key, returning default when any segment is missing."""
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _wrap(raw)
    cfg["_config_path"] = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    return cfg


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config into dotted keys, dropping private entries."""
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).startswith("_"):
                continue
            out.update(flatten(value, f"{prefix}{key}."))
        return out
    if isinstance(data, (list, tuple)):
        out[prefix[:-1]] = ",".join(str(v) for v in data)
        return out
    out[prefix[:-1]] = data
    return out
