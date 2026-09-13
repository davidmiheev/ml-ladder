# F. Geometric & equivariant nets

*ties directly to your algebraic topology / equivariant cohomology background*


**Lab:** [escnn — equivariant CNN library, and PyTorch Geometric tutorials](https://github.com/QUVA-Lab/escnn)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| F1 | Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN | `f1_equivariant_cnn.py` |
| F2 | GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed) | `f2_gcn_graphsage.py` |

Run any rung with `ladder run F1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
