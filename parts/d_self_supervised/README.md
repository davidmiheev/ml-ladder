# D. Self-supervised & representation learning


**Lab:** [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| D1 | SimCLR-style contrastive pretraining on CIFAR-10 | `d1_simclr_pretraining.py` |
| D2 | Masked autoencoder (MAE) on a small image dataset | `d2_masked_autoencoder.py` |
| D3 | Linear probing of the learned representations against a supervised baseline | `d3_linear_probing.py` |

Run any rung with `ladder run D1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
