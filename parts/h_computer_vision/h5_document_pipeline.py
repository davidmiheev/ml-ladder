"""H5. Two-stage document pipeline — detect text regions, then recognize, with end-to-end error attributed to each stage

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run H5`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "h5_document_pipeline",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Two-stage document pipeline — detect text regions, then recognize, with end-to-end error attributed to each stage"""
    todo(
        "H5 — Two-stage document pipeline — detect text regions, then recognize, with end-to-end error attributed to each stage"
    )
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
