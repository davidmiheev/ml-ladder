#!/usr/bin/env python3
"""Create the ``parts/`` tree from the ladder spec.

Idempotent: existing files are left alone, so this is safe to re-run after
adding a rung to ``src/mlladder/ladder.py``. Use ``--check`` in CI to assert the
tree still mirrors the spec 1:1.

    uv run python tools/scaffold.py
    uv run python tools/scaffold.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlladder import ladder as spec
from mlladder.ladder import Item, Part

ROOT = Path(__file__).resolve().parents[1]

STUB_TEMPLATE = '''"""{item_id}. {title}

Lab reference: {lab}

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run {item_id}`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "{rung}",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={{"epochs": 10, "lr": 1e-3, "batch_size": 128}},
)
def run(exp: Experiment) -> dict[str, float]:
    """{title}"""
    todo("{item_id} — {title}")
    return {{}}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
'''

WRITEUP_TEMPLATE = """# Part {part_id} — {title}

> **Status:** not started · TODO(writeup): remove this line when the part is done
> **Numbers:** TODO(writeup): seeds used, e.g. seeds 0-2, reference seed 0 · raw values in [`data/{part_lower}-metrics.json`](data/{part_lower}-metrics.json)
> **Reproduce:** `ladder run {first_code_id}` … , or `uv run python tools/writeup_figures.py {part_id}`
{note}
TODO(writeup): the through-line — the one idea that ties this part's rungs
together, and how it connects to the parts before it.

---
{sections}
## What part {part_id} settled

TODO(writeup): numbered takeaways, including the methodological ones.

---

*Figures regenerate with `uv run python tools/writeup_figures.py {part_id}`; the
numbers they are quoted against live in [`data/{part_lower}-metrics.json`](data/{part_lower}-metrics.json).
See [the writeup conventions](README.md) for the structure this follows.*
"""

WRITEUP_SECTION_TEMPLATE = """
## {item_id} — {short_title}

**Question.** TODO(writeup): what is this rung actually asking?

**What I built.** TODO(writeup): design decisions and the non-obvious details.

**Results.** TODO(writeup): a table with per-seed ranges, and the figures from
`figures/{part_lower}/{item_lower}_*.png`.

**What surprised me.** TODO(writeup)

**Open threads.** TODO(writeup)

**Reading:** TODO(writeup): the entries from [the reading list](../reading-list.md) that belong here.

---
"""

PART_README_TEMPLATE = """# {part_id}. {title}
{note}
{lab}
| Rung | Goal | Module |
| ---- | ---- | ------ |
{rows}

Run any rung with `ladder run {first_id}`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
"""


def rung_slug(item: Item) -> str:
    """Run name for a rung: the module slug, so artifacts/ reads like parts/."""
    assert item.module is not None
    return item.module


def stub_source(part: Part, item: Item) -> str:
    lab = f"[{part.lab.text}]({part.lab.url})" if part.lab else "—"
    return STUB_TEMPLATE.format(item_id=item.id, title=item.title, lab=lab, rung=rung_slug(item))


def part_readme(part: Part) -> str:
    rows = "\n".join(f"| {item.id} | {item.title} | `{item.module}.py` |" for item in part.items if item.module)
    return PART_README_TEMPLATE.format(
        part_id=part.id,
        title=part.title,
        note=f"\n*{part.note}*\n" if part.note else "",
        lab=f"\n**Lab:** [{part.lab.text}]({part.lab.url})\n" if part.lab else "",
        rows=rows,
        first_id=part.items[0].id,
    )


def writeup(part: Part) -> str:
    """Outline for a part's writeup, filled in once its rungs are done."""
    code_items = [item for item in part.items if item.is_code]
    sections = "".join(
        WRITEUP_SECTION_TEMPLATE.format(
            item_id=item.id,
            short_title=item.title.split(" (")[0].split(" — ")[0],
            part_lower=part.id.lower(),
            item_lower=item.id.lower(),
        )
        for item in code_items
    )
    return WRITEUP_TEMPLATE.format(
        part_id=part.id,
        title=part.title,
        part_lower=part.id.lower(),
        first_code_id=code_items[0].id if code_items else part.items[0].id,
        note=f"\n*{part.note}*\n" if part.note else "",
        sections=sections,
    )


def expected_files() -> dict[Path, str]:
    """Every file the spec implies, mapped to the content it would be created with."""
    files: dict[Path, str] = {}
    parts_dir = ROOT / spec.PARTS_DIR_NAME
    files[parts_dir / "__init__.py"] = '"""Ladder rungs, one package per part. Generated from the ladder spec."""\n'
    for part in spec.parts():
        if part.package is None:
            continue
        package_dir = parts_dir / part.package
        files[package_dir / "__init__.py"] = f'"""Part {part.id} — {part.title}."""\n'
        files[package_dir / "README.md"] = part_readme(part)
        for item in part.items:
            if item.module:
                files[package_dir / f"{item.module}.py"] = stub_source(part, item)
        files[ROOT / spec.writeup_path(part)] = writeup(part)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if anything is missing instead of creating it")
    args = parser.parse_args(argv)

    missing: list[Path] = []
    created: list[Path] = []
    for path, content in expected_files().items():
        if path.exists():
            continue
        if args.check:
            missing.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        created.append(path)

    if args.check and missing:
        print("parts/ does not mirror the ladder spec; missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
        return 1
    for path in created:
        print(f"created {path.relative_to(ROOT)}")
    if not created and not args.check:
        print("parts/ already mirrors the ladder spec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
