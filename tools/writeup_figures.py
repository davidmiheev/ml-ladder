#!/usr/bin/env python3
"""Regenerate the figures and headline numbers a part's writeup cites.

Each rung is run once per seed into a scratch directory. The figures from the
reference seed are copied into ``docs/writeups/figures/<part>/`` and every run's
``summary.json`` metrics are collected into ``docs/writeups/data/<part>-metrics.json``,
so a writeup's numbers can be audited (and re-derived) rather than trusted.

    uv run python tools/writeup_figures.py A
    uv run python tools/writeup_figures.py A --seeds 0 1 2 3 4
    uv run python tools/writeup_figures.py A --check   # fail if anything drifted

Reported per metric: the reference-seed value plus the median and spread across
seeds — a claim that only holds on one seed is a claim about that seed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlladder import ladder as spec
from mlladder.ladder import Item, Part

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (0, 1, 2)


def run_rung(item: Item, seed: int, artifacts: Path) -> tuple[Path, dict[str, Any]]:
    """Run one rung into ``artifacts`` and return its run directory and metrics."""
    import importlib
    import os

    os.environ["MLLADDER_ARTIFACTS_DIR"] = str(artifacts)
    os.environ["MLLADDER_TRACKER"] = "jsonl"

    module = importlib.import_module(spec.module_path(item))
    status = module.main(["--seed", str(seed)])
    if status != 0:
        raise SystemExit(f"{item.id} exited with status {status}; writeup figures need a working rung")

    assert item.module is not None
    runs = sorted((artifacts / item.module).iterdir())
    summary = json.loads((runs[-1] / "summary.json").read_text())
    return runs[-1], summary["metrics"]


def sweepable(part: Part, only: list[str] | None) -> list[Item]:
    """The rungs a sweep should run: implemented ones, or an explicit subset.

    An unimplemented rung's stub raises ``NotImplementedError`` by design, so
    sweeping a part mid-build would die on the first stub and produce nothing
    for the rungs that *are* finished. Skipping stubs is therefore the useful
    default — but it is announced rather than silent, because "the sweep
    covered less than you think" is exactly the kind of thing that turns into
    a wrong writeup.
    """
    from mlladder.cli import item_status

    code_items = [item for item in part.items if item.is_code]
    if only:
        wanted = {rung.upper() for rung in only}
        chosen = [item for item in code_items if item.id.upper() in wanted]
        unknown = wanted - {item.id.upper() for item in chosen}
        if unknown:
            raise SystemExit(f"part {part.id} has no rung(s) {sorted(unknown)}")
        return chosen
    ready = [item for item in code_items if item_status(item) == "done"]
    skipped = [item.id for item in code_items if item not in ready]
    if skipped:
        print(f"note: skipping unimplemented {part.id} rungs {skipped}; metrics will cover the rest")
    return ready


def collect(
    part: Part, seeds: tuple[int, ...], reference: int, figures_dir: Path | None, items: list[Item]
) -> dict[str, Any]:
    """Run the given rungs across seeds, copying the reference seed's figures.

    Figures are copied while the scratch directory still exists; ``figures_dir``
    of ``None`` (a --check run) skips them.
    """
    record: dict[str, Any] = {
        "part": part.id,
        "title": part.title,
        "seeds": list(seeds),
        "reference_seed": reference,
        # Recorded so a partial sweep documents itself: a metrics file that
        # covers four of six rungs should say so on its face.
        "swept_rungs": [item.id for item in items],
        "rungs": {},
    }
    copied = 0

    with tempfile.TemporaryDirectory(prefix="mlladder-writeup-") as scratch:
        artifacts = Path(scratch)
        for item in items:
            if not item.is_code:
                continue
            per_seed: dict[str, list[float]] = {}
            for seed in seeds:
                print(f"\n=== {item.id} seed {seed} ===")
                run_dir, metrics = run_rung(item, seed, artifacts)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        per_seed.setdefault(key, []).append(float(value))
                if seed == reference and figures_dir is not None:
                    figures_dir.mkdir(parents=True, exist_ok=True)
                    for figure in sorted(run_dir.glob("*.png")):
                        shutil.copyfile(figure, figures_dir / f"{item.id.lower()}_{figure.name}")
                        copied += 1

            record["rungs"][item.id] = {
                "title": item.title,
                "metrics": {
                    key: {
                        "reference": values[seeds.index(reference)] if len(values) == len(seeds) else values[0],
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "n_seeds": len(values),
                    }
                    for key, values in sorted(per_seed.items())
                    if key != "elapsed_sec"
                },
                "elapsed_sec_median": statistics.median(per_seed.get("elapsed_sec", [0.0])),
            }

    record["figures"] = copied
    return record


# Wall-clock metrics differ between runs on the same seed, so a drift check
# compares everything except these.
TIMING_KEYS = ("seconds", "elapsed", "slowdown", "speedup")


def deterministic_part(record: dict[str, Any]) -> dict[str, Any]:
    """The metrics that must reproduce exactly for a given seed."""
    return {
        rung_id: {key: stats for key, stats in rung["metrics"].items() if not any(word in key for word in TIMING_KEYS)}
        for rung_id, rung in record["rungs"].items()
    }


def markdown_table(record: dict[str, Any]) -> str:
    """A pasteable table of every metric, for writing the numbers into prose."""
    lines = []
    for rung_id, rung in record["rungs"].items():
        lines.append(f"\n### {rung_id} — {rung['title']}\n")
        lines.append("| metric | seed 0 | median | range |")
        lines.append("| --- | --- | --- | --- |")
        for key, stats in rung["metrics"].items():
            spread = f"{stats['min']:.4g} – {stats['max']:.4g}" if stats["n_seeds"] > 1 else "—"
            lines.append(f"| `{key}` | {stats['reference']:.4g} | {stats['median']:.4g} | {spread} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", help="part ids to regenerate, e.g. A B")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--reference", type=int, default=0, help="seed whose figures are kept (default: 0)")
    parser.add_argument(
        "--check", action="store_true", help="regenerate into a scratch dir and diff against what is committed"
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        metavar="ID",
        help="sweep only these rungs (e.g. --rungs D1); default is every implemented rung of the part",
    )
    args = parser.parse_args(argv)

    if args.reference not in args.seeds:
        parser.error(f"--reference {args.reference} must be one of --seeds {args.seeds}")

    for part_id in args.parts:
        part = spec.get_part(part_id)
        figures_dir = None if args.check else ROOT / spec.figures_dir(part)
        items = sweepable(part, args.rungs)
        if not items:
            print(f"part {part.id} has no implemented rungs to sweep", file=sys.stderr)
            return 1
        record = collect(part, tuple(args.seeds), args.reference, figures_dir, items)

        metrics_file = ROOT / spec.metrics_path(part)
        if args.check:
            if not metrics_file.exists():
                print(f"{spec.metrics_path(part)} is missing — run tools/writeup_figures.py {part.id}", file=sys.stderr)
                return 1
            existing = deterministic_part(json.loads(metrics_file.read_text()))
            fresh = deterministic_part(record)
            if existing != fresh:
                changed = sorted(
                    f"{rung}.{key}"
                    for rung in fresh
                    for key in set(fresh[rung]) | set(existing.get(rung, {}))
                    if existing.get(rung, {}).get(key) != fresh[rung].get(key)
                )
                print(f"{spec.metrics_path(part)} is stale — rerun tools/writeup_figures.py {part.id}", file=sys.stderr)
                print("  changed: " + ", ".join(changed), file=sys.stderr)
                return 1
            print(f"{part.id}: metrics match what is committed (timings excluded)")
            continue

        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if args.rungs and metrics_file.exists():
            # Merge into what is there, or `--rungs D1` would delete D2 and D3
            # from the file and quietly shrink the writeup's evidence base.
            existing_record = json.loads(metrics_file.read_text())
            existing_record["rungs"].update(record["rungs"])
            existing_record["swept_rungs"] = sorted(
                set(existing_record.get("swept_rungs", [])) | set(record["swept_rungs"])
            )
            existing_record["figures"] = record["figures"]
            record = existing_record
        metrics_file.write_text(json.dumps(record, indent=2) + "\n")

        print(f"\nwrote {record['figures']} figures to {spec.figures_dir(part)}/ and {spec.metrics_path(part)}")
        print(markdown_table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
