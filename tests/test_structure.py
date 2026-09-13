"""The folder tree must mirror the ladder spec 1:1 (rung 01)."""

from __future__ import annotations

import importlib

import pytest

from mlladder import ladder as spec
from mlladder.cli import item_status, module_file

CODE_ITEMS = [item for item in spec.iter_items() if item.is_code]


def test_every_part_has_items():
    for part in spec.parts():
        assert part.items, f"part {part.id} has no rungs"


def test_item_ids_are_unique_and_prefixed_by_part():
    seen = set()
    for part in spec.parts():
        for item in part.items:
            assert item.id not in seen, f"duplicate rung id {item.id}"
            seen.add(item.id)
            assert item.id.startswith(part.id), f"{item.id} does not belong to part {part.id}"


@pytest.mark.parametrize("item", CODE_ITEMS, ids=lambda item: item.id)
def test_code_item_has_a_module_file(item):
    assert module_file(item).exists(), f"missing module for {item.id}; run tools/scaffold.py"


@pytest.mark.parametrize("item", CODE_ITEMS, ids=lambda item: item.id)
def test_code_item_is_importable_and_has_main(item):
    module = importlib.import_module(spec.module_path(item))
    assert callable(module.main), f"{item.id} has no callable main()"


@pytest.mark.parametrize("item", CODE_ITEMS, ids=lambda item: item.id)
def test_status_is_known(item):
    assert item_status(item) in {"done", "stub"}


def test_setup_part_has_no_modules():
    assert all(not item.is_code for item in spec.get_part("0").items)


def test_lookup_helpers_are_case_insensitive():
    assert spec.get_item("b2").id == "B2"
    assert spec.get_part("g").title == "Scaling & systems"
    with pytest.raises(KeyError):
        spec.get_item("Z9")
