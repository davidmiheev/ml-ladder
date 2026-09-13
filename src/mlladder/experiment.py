"""The entry point every rung uses.

A rung module defines ``run(exp)`` and ends with ``main = experiment(...)``; the
harness then gives it argument parsing, seeding, a run directory, tracking and a
saved summary, so the rung file only contains the ML.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlladder import paths, plotting
from mlladder.seeding import seed_everything
from mlladder.tracking import Tracker, make_tracker


@dataclass
class Experiment:
    """Everything a rung needs from the harness, handed to ``run(exp)``."""

    name: str
    config: dict[str, Any]
    run_dir: Path
    tracker: Tracker
    seed: int
    metrics: dict[str, Any] = field(default_factory=dict)

    # -- config access -----------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    # -- logging -----------------------------------------------------------
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.tracker.log(metrics, step)

    def log_figure(self, name: str, figure: Any) -> Path:
        """Log a matplotlib figure to the tracker and save it in the run directory."""
        self.tracker.log_figure(name, figure)
        return plotting.save_fig(figure, self.run_dir / f"{name}.png")

    def record(self, **values: Any) -> None:
        """Record headline numbers; written to ``summary.json`` at the end."""
        self.metrics.update(values)
        self.tracker.summary(values)

    def path(self, *parts: str) -> Path:
        """A path inside this run's directory, with parents created."""
        target = self.run_dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def data_dir(self) -> Path:
        return paths.data_dir()


def experiment(
    name: str,
    *,
    defaults: dict[str, Any] | None = None,
    description: str | None = None,
    tags: Sequence[str] = (),
) -> Callable[..., int]:
    """Wrap ``run(exp)`` into a ``main(argv)`` with CLI, seeding and tracking.

    ``defaults`` becomes both the config and the CLI flags: ``{"lr": 1e-3}``
    gives ``--lr`` parsed as a float. Types are taken from the default values,
    so keep them concrete.
    """
    defaults = dict(defaults or {})

    def decorate(run: Callable[[Experiment], Any]) -> Callable[..., int]:
        def main(argv: Sequence[str] | None = None) -> int:
            parser = argparse.ArgumentParser(prog=name, description=description or run.__doc__)
            parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
            parser.add_argument(
                "--track",
                choices=("auto", "wandb", "jsonl", "none"),
                default=None,
                help="run-tracking backend (default: auto — wandb when configured, else jsonl)",
            )
            parser.add_argument("--run-name", default=None, help="override the run directory name")
            for key, value in defaults.items():
                _add_config_argument(parser, key, value)

            args = parser.parse_args(argv)
            config = {_dest(key): getattr(args, _dest(key)) for key in defaults}

            seed = seed_everything(args.seed)
            run_dir = paths.run_dir(args.run_name or name)
            plotting.use_house_style()

            full_config = {**config, "seed": seed, "rung": name}
            tracker = make_tracker(
                args.run_name or name,
                run_dir=run_dir,
                config=full_config,
                backend=args.track,
                tags=[name.split("_")[0], *tags],
            )
            exp = Experiment(name=name, config=config, run_dir=run_dir, tracker=tracker, seed=seed)

            print(f"[{name}] run dir: {run_dir}")
            started = time.perf_counter()
            status = "ok"
            try:
                result = run(exp)
            except NotImplementedError as error:
                status = "not-implemented"
                print(f"[{name}] {error}")
                return 2
            except KeyboardInterrupt:
                status = "interrupted"
                print(f"[{name}] interrupted")
                return 130
            else:
                if isinstance(result, dict):
                    exp.record(**result)
                return 0
            finally:
                elapsed = time.perf_counter() - started
                exp.metrics.setdefault("status", status)
                exp.metrics.setdefault("elapsed_sec", round(elapsed, 3))
                _write_summary(exp, full_config)
                tracker.finish()
                print(f"[{name}] {status} in {elapsed:.2f}s")

        main.__name__ = "main"
        main.__doc__ = run.__doc__
        main.run = run  # type: ignore[attr-defined]
        main.rung = name  # type: ignore[attr-defined]
        return main

    return decorate


def _dest(key: str) -> str:
    return key.replace("-", "_")


def _add_config_argument(parser: argparse.ArgumentParser, key: str, value: Any) -> None:
    flag = f"--{key.replace('_', '-')}"
    dest = _dest(key)
    if isinstance(value, bool):
        parser.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction, default=value)
    elif isinstance(value, (list, tuple)):
        element = type(value[0]) if value else str
        parser.add_argument(flag, dest=dest, nargs="+", type=element, default=list(value))
    else:
        parser.add_argument(flag, dest=dest, type=type(value) if value is not None else str, default=value)


def _write_summary(exp: Experiment, config: dict[str, Any]) -> None:
    summary = {
        "rung": exp.name,
        "config": config,
        "metrics": exp.metrics,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv[1:],
    }
    (exp.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


def todo(what: str) -> None:
    """Marker for an unimplemented rung — keeps the stub runnable and honest."""
    raise NotImplementedError(f"not implemented yet: {what}")
