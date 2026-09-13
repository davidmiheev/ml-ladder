"""Repo-relative paths. Everything writable lives under ``artifacts/`` or ``data/``."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


def repo_root() -> Path:
    """Walk up from this file until a repo marker shows up."""
    for candidate in Path(__file__).resolve().parents:
        if any((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    return Path.cwd()


def _dir(env_var: str, default: str) -> Path:
    override = os.environ.get(env_var)
    path = Path(override).expanduser() if override else repo_root() / default
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Downloaded datasets. Override with ``MLLADDER_DATA_DIR`` to share a cache."""
    return _dir("MLLADDER_DATA_DIR", "data")


def artifacts_dir() -> Path:
    """Run outputs: figures, metrics, checkpoints."""
    return _dir("MLLADDER_ARTIFACTS_DIR", "artifacts")


def run_dir(name: str, *, timestamp: bool = True) -> Path:
    """A fresh directory for one run's outputs."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = artifacts_dir() / name / stamp if timestamp else artifacts_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
