"""Writeups must exist, and a finished part must have a finished writeup.

The point of these tests is the last one: it is easy to implement a part and
move on, so CI fails when a part's rungs are all done and its writeup still
carries the scaffold's TODO markers.
"""

from __future__ import annotations

import json
import re

import pytest

from mlladder import ladder as spec
from mlladder.cli import item_status
from mlladder.paths import repo_root

ROOT = repo_root()
DOCUMENTED_PARTS = [part for part in spec.parts() if part.package is not None]
TODO_MARKER = "TODO(writeup)"


def part_is_complete(part) -> bool:
    return all(item_status(item) == "done" for item in part.items if item.is_code)


COMPLETE_PARTS = [part for part in DOCUMENTED_PARTS if part_is_complete(part)]


@pytest.mark.parametrize("part", DOCUMENTED_PARTS, ids=lambda part: part.id)
def test_every_part_has_a_writeup(part):
    path = ROOT / spec.writeup_path(part)
    assert path.exists(), f"missing writeup for part {part.id}; run tools/scaffold.py"
    text = path.read_text()
    assert text.startswith(f"# Part {part.id} —")
    for item in part.items:
        if item.is_code:
            assert f"## {item.id} —" in text, f"writeup for part {part.id} has no section for {item.id}"


@pytest.mark.parametrize("part", COMPLETE_PARTS, ids=lambda part: part.id)
def test_completed_parts_have_finished_writeups(part):
    """Every rung implemented means the writeup is due — no TODO markers left."""
    text = (ROOT / spec.writeup_path(part)).read_text()
    assert TODO_MARKER not in text, f"part {part.id} is finished but its writeup still has {TODO_MARKER} markers"
    assert "**Status:** complete" in text, f"part {part.id}'s writeup is not marked complete"
    assert "What surprised me" in text


@pytest.mark.parametrize("part", COMPLETE_PARTS, ids=lambda part: part.id)
def test_writeup_figures_exist(part):
    """Every image a writeup references must be committed next to it."""
    writeup = ROOT / spec.writeup_path(part)
    references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", writeup.read_text())
    assert references, f"part {part.id}'s writeup has no figures"
    for reference in references:
        assert (writeup.parent / reference).exists(), (
            f"{reference} is referenced but missing; rerun tools/writeup_figures.py {part.id}"
        )


@pytest.mark.parametrize("part", COMPLETE_PARTS, ids=lambda part: part.id)
def test_metrics_file_covers_every_rung(part):
    """The numbers a writeup quotes come from this file, so it must be current."""
    path = ROOT / spec.metrics_path(part)
    assert path.exists(), f"missing {spec.metrics_path(part)}; run tools/writeup_figures.py {part.id}"
    record = json.loads(path.read_text())

    assert record["part"] == part.id
    assert len(record["seeds"]) >= 3, "headline numbers need at least three seeds"
    assert record["reference_seed"] in record["seeds"]
    for item in part.items:
        if item.is_code:
            assert item.id in record["rungs"], f"{item.id} has no metrics recorded"
            assert record["rungs"][item.id]["metrics"], f"{item.id} recorded no metrics"


def test_writeup_index_lists_every_part():
    index = (ROOT / spec.WRITEUPS_DIR / "README.md").read_text()
    for part in DOCUMENTED_PARTS:
        assert f"]({spec.part_slug(part)}.md)" in index, f"part {part.id} is missing from the writeup index"
