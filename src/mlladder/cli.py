"""``ladder`` command line: list the rungs, check progress, run one.

ladder list                 # every part and rung
ladder status               # which rungs are still stubs
ladder run B2 --epochs 5    # run a rung, passing flags straight through
ladder docs                 # regenerate docs/ladder.md from the spec
"""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from mlladder import ladder as spec
from mlladder.ladder import Item, Part

# A rung still calling ``todo(...)`` has not been implemented yet. Checked by
# reading the source rather than importing, so ``status`` stays fast and needs
# none of the heavy optional dependencies.
_TODO_MARKERS = ("todo(", "NotImplementedError")


def module_file(item: Item) -> Path:
    part = spec.get_part(item.part_id)
    assert part.package is not None and item.module is not None
    return spec_root() / spec.PARTS_DIR_NAME / part.package / f"{item.module}.py"


def spec_root() -> Path:
    from mlladder.paths import repo_root

    return repo_root()


def item_status(item: Item) -> str:
    """``done``, ``stub``, ``missing`` — or ``setup`` for the part-0 rungs."""
    if not item.is_code:
        return "setup"
    path = module_file(item)
    if not path.exists():
        return "missing"
    source = path.read_text()
    return "stub" if any(marker in source for marker in _TODO_MARKERS) else "done"


_SYMBOLS = {"done": "[x]", "stub": "[ ]", "missing": "[!]", "setup": "[x]"}


def _format_item(item: Item, status: str) -> str:
    return f"  {_SYMBOLS[status]} {item.id:<3} {item.title}"


def cmd_list(args: argparse.Namespace) -> int:
    for part in spec.parts():
        if args.part and part.id != args.part.upper():
            continue
        _print_part_header(part)
        for item in part.items:
            print(_format_item(item, item_status(item)))
        print()
    return 0


def _print_part_header(part: Part) -> None:
    header = f"{part.id}. {part.title}"
    print(header)
    print("-" * len(header))
    if part.note:
        print(f"  note: {part.note}")
    if part.lab:
        print(f"  lab:  {part.lab.text} — {part.lab.url}")


def cmd_status(args: argparse.Namespace) -> int:
    if args.complete:
        # Machine-readable: the parts whose code rungs are all implemented.
        print(" ".join(part.id for part in spec.parts() if part.package and _part_is_complete(part)))
        return 0

    total = implemented = 0
    for part in spec.parts():
        counts = {"done": 0, "stub": 0, "missing": 0, "setup": 0}
        for item in part.items:
            counts[item_status(item)] += 1
        part_done = counts["done"] + counts["setup"]
        part_total = len(part.items)
        total += part_total
        implemented += part_done
        bar = _bar(part_done, part_total)
        missing = f"  ({counts['missing']} missing)" if counts["missing"] else ""
        print(f"{part.id}  {bar}  {part_done}/{part_total}  {part.title}{missing}")
    print(f"\ntotal: {implemented}/{total} rungs done ({100 * implemented // total}%)")
    return 0


def _part_is_complete(part: Part) -> bool:
    return all(item_status(item) == "done" for item in part.items if item.is_code)


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * filled + "·" * (width - filled)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        item = spec.get_item(args.item)
    except KeyError as error:
        print(error, file=sys.stderr)
        return 1
    if not item.is_code:
        print(f"{item.id} is a setup rung — see docs/ladder.md, nothing to run.", file=sys.stderr)
        return 1

    path = spec.module_path(item)
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError as error:
        print(f"cannot import {path}: {error}", file=sys.stderr)
        return 1

    main = getattr(module, "main", None)
    if main is None:
        # Not harness-wrapped: run it as a script so ad-hoc rungs still work.
        sys.argv = [path, *args.rest]
        runpy.run_module(path, run_name="__main__")
        return 0
    return int(main(args.rest) or 0)


def cmd_docs(args: argparse.Namespace) -> int:
    from mlladder.docgen import write_ladder_doc

    destination = write_ladder_doc(Path(args.out) if args.out else None)
    print(f"wrote {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ladder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list parts and rungs")
    listing.add_argument("part", nargs="?", help="restrict to one part, e.g. B")
    listing.set_defaults(func=cmd_list)

    status = sub.add_parser("status", help="progress per part")
    status.add_argument("--complete", action="store_true", help="print just the ids of parts whose rungs are all done")
    status.set_defaults(func=cmd_status)

    run = sub.add_parser("run", help="run one rung by id, e.g. ladder run B2 --epochs 5")
    run.add_argument("item", help="rung id, e.g. A1")
    run.add_argument("rest", nargs=argparse.REMAINDER, help="arguments forwarded to the rung")
    run.set_defaults(func=cmd_run)

    docs = sub.add_parser("docs", help="regenerate docs/ladder.md from the spec")
    docs.add_argument("--out", default=None, help="output path (default: docs/ladder.md)")
    docs.set_defaults(func=cmd_docs)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        # `ladder status | head` closes the pipe early; that is not an error.
        # Redirect stdout to devnull so Python's shutdown flush stays quiet too.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
