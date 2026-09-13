"""Run tracking with a W&B backend and a JSONL fallback.

The fallback matters more than the backend: every rung logs through ``Tracker``,
so a run works identically with no wandb installed, no API key, or no network.
Set ``MLLADDER_TRACKER=wandb|jsonl|none`` to force a backend; the default picks
wandb when it is importable and configured, and JSONL otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

WANDB_PROJECT_ENV = "WANDB_PROJECT"
DEFAULT_PROJECT = "ml-ladder"


class Tracker(Protocol):
    """What the rungs are allowed to assume about a tracking backend."""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def log_figure(self, name: str, figure: Any) -> None: ...

    def summary(self, values: dict[str, Any]) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Drops everything. Used by tests and ``--track none``."""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_figure(self, name: str, figure: Any) -> None:
        return None

    def summary(self, values: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class JSONLTracker:
    """Appends one JSON object per ``log`` call to ``<run_dir>/metrics.jsonl``.

    Headline numbers go to ``metrics_summary.json`` (the harness writes the
    canonical ``summary.json``), and figures are saved next to them, so a local
    run leaves the same artifacts a W&B run would show.
    """

    def __init__(self, run_dir: Path, config: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._summary_path = self.run_dir / "metrics_summary.json"
        self._summary: dict[str, Any] = {}
        self._step = 0
        if config:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step + 1
        record = {"step": step, **{k: _jsonable(v) for k, v in metrics.items()}}
        with self._metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_figure(self, name: str, figure: Any) -> None:
        from mlladder.plotting import save_fig

        save_fig(figure, self.run_dir / f"{name}.png", close=False)

    def summary(self, values: dict[str, Any]) -> None:
        self._summary.update({k: _jsonable(v) for k, v in values.items()})
        self._summary_path.write_text(json.dumps(self._summary, indent=2) + "\n")

    def finish(self) -> None:
        if self._summary:
            self._summary_path.write_text(json.dumps(self._summary, indent=2) + "\n")


class WandbTracker:
    """Thin wrapper over ``wandb.run``; mirrors metrics into the run directory too."""

    def __init__(self, run: Any, mirror: JSONLTracker | None = None) -> None:
        self._run = run
        self._mirror = mirror

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log({k: _jsonable(v) for k, v in metrics.items()}, step=step)
        if self._mirror:
            self._mirror.log(metrics, step)

    def log_figure(self, name: str, figure: Any) -> None:
        import wandb

        self._run.log({name: wandb.Image(figure)})
        if self._mirror:
            self._mirror.log_figure(name, figure)

    def summary(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self._run.summary[key] = _jsonable(value)
        if self._mirror:
            self._mirror.summary(values)

    def finish(self) -> None:
        self._run.finish()
        if self._mirror:
            self._mirror.finish()


def _jsonable(value: Any) -> Any:
    """Unwrap the usual suspects (numpy scalars, torch tensors) into JSON types."""
    for attr in ("item", "tolist"):
        if hasattr(value, attr) and not isinstance(value, (str, bytes, int, float, bool)):
            try:
                return getattr(value, attr)()
            except (ValueError, TypeError, RuntimeError):
                continue
    return value


def wandb_available() -> bool:
    """True when wandb is importable and has credentials (or is set to offline)."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    mode = os.environ.get("WANDB_MODE", "")
    if mode in {"offline", "dryrun", "disabled"}:
        return mode != "disabled"
    return bool(os.environ.get("WANDB_API_KEY")) or (Path.home() / ".netrc").exists()


def make_tracker(
    name: str,
    *,
    run_dir: Path,
    config: dict[str, Any] | None = None,
    backend: str | None = None,
    tags: list[str] | None = None,
) -> Tracker:
    """Build the tracker for a run, falling back rather than failing."""
    choice = (backend or os.environ.get("MLLADDER_TRACKER") or "auto").lower()
    if choice == "none":
        return NullTracker()
    if choice in {"auto", "wandb"} and wandb_available():
        try:
            import wandb

            run = wandb.init(
                project=os.environ.get(WANDB_PROJECT_ENV, DEFAULT_PROJECT),
                name=name,
                config=config or {},
                tags=tags or [],
                dir=str(run_dir),
            )
            return WandbTracker(run, mirror=JSONLTracker(run_dir, config))
        except Exception as error:  # tracking must never fail a run
            print(f"[tracking] wandb unavailable ({error}); falling back to JSONL")
    elif choice == "wandb":
        print("[tracking] wandb requested but not configured; falling back to JSONL")
    return JSONLTracker(run_dir, config)
