# H. Computer vision & OCR

*follows C; independent of D-G, so it can be taken straight after the ViT rung*


**Lab:** [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| H1 | Object detection from scratch — IoU, non-max suppression and mAP implemented by hand | `h1_detection_from_scratch.py` |
| H2 | Semantic segmentation with a U-Net — skip connections, Dice vs cross-entropy loss | `h2_unet_segmentation.py` |
| H3 | Text recognition with CRNN + CTC — synthetic text lines, greedy vs beam decoding, CER/WER | `h3_crnn_ctc.py` |
| H4 | Attention-decoder OCR against CTC on the same data, with a pretrained model as the baseline | `h4_attention_ocr.py` |
| H5 | Two-stage document pipeline — detect text regions, then recognize, with end-to-end error attributed to each stage | `h5_document_pipeline.py` |
| H6 | OCR as a vision-language model — PaddleOCR-VL and Unlimited-OCR read from source, against the H5 pipeline | `h6_ocr_vlm_architectures.py` |

Run any rung with `ladder run H1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.

## What the sweep found

Medians over seeds 0–2; H1–H5 on a Colab T4, H6 is arithmetic. The full
argument, with derivations, is in the writeup.

**H1 — the metric is a construction.** The same predictions score 0.955 at IoU
0.5 and **0.650 at 0.75**. The NMS dial moves mAP@0.5 by 0.109 and mAP@0.75 by
only 0.032: at the strict threshold, localization binds and duplicate removal
stops mattering.

**H2 — score where the mechanism acts.** Skip connections are worth +0.013
foreground mIoU overall and **+0.030 on a 2px boundary band** — 2.3× more where
the theory says they act. Whole-image mIoU was averaging that away. Dice vs
cross-entropy does *not* survive three seeds (wins twice, loses once).

**H3 — decoding was never the bottleneck.** Beam-15 over the same posteriors
gains **0.0000 CER**; held-out fonts cost 0.164. And that gap is one typeface:

| held-out font | CER |
| --- | --- |
| DejaVuSansMono-Oblique | 0.0038 |
| DejaVuSerif-Italic | 0.0052 |
| cmr10 | **0.4666** |

Two of three held-out faces are DejaVu family members of training faces, so the
aggregate hid a near-perfect result and a hard one.

**H4 — which decoder wins depends on the metric.** On held-out fonts CTC wins
CER (0.176 vs 0.196) and attention wins WER (0.260 vs 0.409) and exact match
(0.701 vs 0.613). CTC is exactly length-invariant (+0.0003); the attention
decoder's +0.220 length gap is **+0.139 canvas width** and only +0.081 the extra
word. Zero-shot TrOCR's font gap is −0.001.

**H5 — the interface is the error.** Detection is **99.98%** of page-level CER
on every seed. The recognizer breaks at ±0.33px of crop error; jitter
augmentation makes it 24× more tolerant for free and still recovers only a sixth
of the pipeline's error, because it cannot read a line that was never found.

**H6 — the cost of deleting the interface.** PaddleOCR-VL is 904M parameters
(45.5% vision) against this pipeline's 488k; Unlimited-OCR's text tower holds
2.93B and runs 740M per token. A 384px page is 729 patches → 182 tokens after
the 2×2 merge, 16× cheaper in attention. Parameter formulas validated against
TrOCR to 1.3%.
