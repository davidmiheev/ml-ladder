"""F2. GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed)

Lab reference: [escnn — equivariant CNN library, and PyTorch Geometric tutorials](https://github.com/QUVA-Lab/escnn)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run F2`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "f2_gcn_graphsage",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed)"""
    todo("F2 — GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed)")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
