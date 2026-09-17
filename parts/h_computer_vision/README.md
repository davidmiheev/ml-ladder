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
