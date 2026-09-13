"""The shared harness: experiments, tracking, metrics, plotting."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mlladder import metrics, plotting
from mlladder.experiment import Experiment, experiment
from mlladder.tracking import JSONLTracker, NullTracker, make_tracker


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Keep runs out of the repo's artifacts/ and data/ directories."""
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MLLADDER_TRACKER", "jsonl")


def test_experiment_runs_and_writes_a_summary():
    @experiment("test_rung", defaults={"epochs": 2, "lr": 0.5, "batch_size": 8})
    def run(exp: Experiment) -> dict[str, float]:
        assert exp["epochs"] == 3  # overridden on the command line below
        assert exp.get("missing", "fallback") == "fallback"
        for step in range(exp["epochs"]):
            exp.log({"loss": 1.0 / (step + 1)}, step=step)
        return {"final_loss": 0.25}

    assert run(["--epochs", "3", "--seed", "7"]) == 0

    run_dirs = list((Path(os.environ["MLLADDER_ARTIFACTS_DIR"]) / "test_rung").iterdir())
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text())
    assert summary["config"]["epochs"] == 3
    assert summary["config"]["seed"] == 7
    assert summary["metrics"]["final_loss"] == 0.25
    assert summary["metrics"]["status"] == "ok"

    logged = [json.loads(line) for line in (run_dirs[0] / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in logged] == [0, 1, 2]


def test_stub_rung_exits_with_code_two():
    from mlladder.experiment import todo

    @experiment("stub_rung")
    def run(exp: Experiment) -> dict[str, float]:
        todo("nothing here yet")
        return {}

    assert run([]) == 2


def test_boolean_and_list_flags():
    seen = {}

    @experiment("flags_rung", defaults={"augment": True, "layers": [1, 2]})
    def run(exp: Experiment) -> None:
        seen.update(exp.config)

    assert run(["--no-augment", "--layers", "3", "4"]) == 0
    assert seen == {"augment": False, "layers": [3, 4]}


def test_seeding_is_reproducible():
    from mlladder.seeding import seed_everything

    seed_everything(123)
    first = np.random.rand(4)
    seed_everything(123)
    assert np.allclose(first, np.random.rand(4))


def test_jsonl_tracker_writes_config_metrics_and_summary(tmp_path):
    tracker = JSONLTracker(tmp_path / "run", config={"lr": 0.1})
    tracker.log({"loss": np.float32(0.5)})
    tracker.summary({"best": np.float64(0.9)})
    tracker.finish()

    assert json.loads((tmp_path / "run" / "config.json").read_text()) == {"lr": 0.1}
    record = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().strip())
    assert record == {"step": 0, "loss": pytest.approx(0.5)}
    assert json.loads((tmp_path / "run" / "metrics_summary.json").read_text())["best"] == pytest.approx(0.9)


def test_tracker_backend_selection(tmp_path):
    assert isinstance(make_tracker("r", run_dir=tmp_path, backend="none"), NullTracker)
    assert isinstance(make_tracker("r", run_dir=tmp_path, backend="jsonl"), JSONLTracker)
    # wandb is not configured in CI, so "wandb" must degrade rather than fail.
    assert isinstance(make_tracker("r", run_dir=tmp_path, backend="wandb"), JSONLTracker)


def test_classification_metrics():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 1, 2, 1, 1]
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(0.8)
    matrix = metrics.confusion_matrix(y_true, y_pred)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 5
    scores = metrics.precision_recall_f1(y_true, y_pred)
    assert scores["recall"][1] == pytest.approx(1.0)
    assert scores["recall"][2] == pytest.approx(0.5)
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert metrics.cross_entropy([0, 1], probs) == pytest.approx(0.16425, abs=1e-4)
    assert metrics.top_k_accuracy([0, 1], probs, k=1) == pytest.approx(1.0)


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    assert metrics.mse(y_true, y_pred) == pytest.approx(0.16667, abs=1e-4)
    assert metrics.rmse(y_true, y_pred) == pytest.approx(0.40825, abs=1e-4)
    assert metrics.mae(y_true, y_pred) == pytest.approx(0.33333, abs=1e-4)
    assert metrics.r2_score(y_true, y_pred) == pytest.approx(0.75)
    assert metrics.r2_score([2.0, 2.0], [2.0, 2.0]) == 0.0  # zero-variance guard


def test_plot_helpers_write_files(tmp_path):
    curves = plotting.plot_curves({"train": [1.0, 0.5], "val": [1.1, 0.6]}, title="loss")
    assert plotting.save_fig(curves, tmp_path / "curves.png").exists()

    confusion = plotting.plot_confusion_matrix(np.eye(3, dtype=int), labels=["a", "b", "c"], normalize=True)
    assert plotting.save_fig(confusion, tmp_path / "cm.png").exists()

    bars = plotting.plot_comparison({"sgd": 0.9, "adam": 0.93}, ylabel="accuracy")
    assert plotting.save_fig(bars, tmp_path / "bars.png").exists()

    grid = plotting.plot_image_grid(np.random.rand(6, 1, 8, 8), nrow=3, title="samples")
    assert plotting.save_fig(grid, tmp_path / "grid.png").exists()
