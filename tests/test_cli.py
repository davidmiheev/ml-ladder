"""The ``ladder`` command line."""

from __future__ import annotations

import pytest

from mlladder import cli, docgen
from mlladder import ladder as spec


def test_list_prints_every_rung(capsys):
    assert cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "A1" in output and "G3" in output
    assert "Classical ML from scratch" in output


def test_list_can_restrict_to_a_part(capsys):
    assert cli.main(["list", "f"]) == 0
    output = capsys.readouterr().out
    assert "F1" in output
    assert "A1" not in output


def test_status_reports_a_total(capsys):
    assert cli.main(["status"]) == 0
    assert "total:" in capsys.readouterr().out


def test_run_rejects_unknown_and_setup_rungs(capsys):
    assert cli.main(["run", "Z9"]) == 1
    assert cli.main(["run", "01"]) == 1


def test_run_reports_an_unwritten_rung(tmp_path, monkeypatch):
    """A rung that is still a stub runs, says so, and exits 2."""
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")

    stub = next((item for item in spec.iter_items() if cli.item_status(item) == "stub"), None)
    if stub is None:
        pytest.skip("every rung is implemented")
    assert cli.main(["run", stub.id]) == 2


def test_docs_are_generated_from_the_spec(tmp_path):
    destination = docgen.write_ladder_doc(tmp_path / "ladder.md")
    text = destination.read_text()
    assert "## A. Classical ML from scratch" in text
    assert "parts/a_classical/a1_linear_logistic_regression.py" in text


def test_committed_docs_match_the_spec():
    """docs/ladder.md is generated; regenerate with `ladder docs` after editing the spec."""
    from mlladder.paths import repo_root

    committed = repo_root() / "docs" / "ladder.md"
    assert committed.read_text() == docgen.render(), "docs/ladder.md is stale — run `ladder docs`"
