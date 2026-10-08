"""transformers' tiny.en with timestamps on, over the same clips: the reference
the candle example's timestamp rules are compared against.

usage: python whisper_hf_timestamps.py CLIPS_DIR OUT_JSON
OUT_JSON must already hold an "upstream" arm from whisper_example_rules.py,
whose clip order and ground truth are reused.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

clips_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = json.loads(out.read_text())["upstream"]["rows"]  # same clip order and truth
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval()
rows = []
for r in spec:
    with wave.open(str(clips_dir / r["clip"])) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    feats = processor(x, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats, return_timestamps=True, max_new_tokens=200)
    toks = ids[0].tolist()
    if toks[0] != model.config.decoder_start_token_id:  # generate may drop the prompt
        toks = [model.config.decoder_start_token_id, *toks]
    text = processor.decode(ids[0], skip_special_tokens=True)
    rows.append(
        {"clip": r["clip"], "kind": r["kind"], "words": r["words"], "tokens": [toks], "text": text.strip(), "spans": []}
    )
    print(r["clip"], repr(text[:60]), toks[:6], flush=True)
report = json.loads(out.read_text())
report["hf-timestamps"] = {"rows": rows}
out.write_text(json.dumps(report, indent=1))
