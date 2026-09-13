# G. Scaling & systems

*bridges into ignite-catalog*


**Lab:** [PyTorch mixed-precision & profiler docs](https://pytorch.org/docs/stable/notes/amp_examples.html)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| G1 | Fit your own scaling-law curve (loss vs. params/data/compute) on a small model family | `g1_scaling_law_fit.py` |
| G2 | Mixed-precision + gradient-accumulation training, profiled for throughput and memory | `g2_mixed_precision_profiling.py` |
| G3 | Rust/Candle port of one transformer-block forward pass, as a comprehension exercise for [[ignite-catalog]] | `g3_candle_transformer_port.py` |

Run any rung with `ladder run G1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
