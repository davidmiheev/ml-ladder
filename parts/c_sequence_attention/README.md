# C. Sequence models & attention


**Lab:** [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| C1 | Char-level RNN/LSTM language model | `c1_char_rnn.py` |
| C2 | Seq2seq with attention on a toy translation task | `c2_seq2seq_attention.py` |
| C3 | Transformer from scratch (attention, positional encoding, causal masking) | `c3_transformer_from_scratch.py` |
| C4 | Fine-tune a small ViT on a custom image-classification task | `c4_vit_finetune.py` |

Run any rung with `ladder run C1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
