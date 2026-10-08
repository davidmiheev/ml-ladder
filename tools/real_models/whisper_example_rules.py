"""Run candle's whisper example over the I-part clips and score it the way the executor study did.

usage: [WMODEL=tiny] python whisper_example_rules.py BINARY CLIPS_DIR LABEL OUT_JSON [EXTRA EXAMPLE ARGS...]

CLIPS_DIR holds the I-part clips at 16 kHz (the example rejects other rates).
Each run adds one arm, keyed by LABEL, to OUT_JSON. The arms recorded in
docs/writeups/data/i-whisper-example-rules.json were made with this script.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

binary, clips_dir, label, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
extra = sys.argv[5:]
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
NUMERALS = {str(i): WORDS[i] for i in range(10)}
HOMOPHONES = {"to": "two", "too": "two", "for": "four", "won": "one", "ate": "eight", "oh": "zero"}


def words_of(text):
    out = []
    for token in re.findall(r"[a-z0-9']+", text.lower()):
        if token.isdigit() and len(token) > 1:
            out += [WORDS[int(c)] for c in token]
        else:
            out.append(NUMERALS.get(token, token))
    return out


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return d[len(hyp)] / max(len(ref), 1)


def empty_segments(tokens):
    """Timestamp-delimited segments with no text tokens, counting an opening
    timestamp followed directly by end-of-text. Token ids are read off the
    prompt: <|startoftranscript|> is end-of-text + 1 and the first timestamp
    <|0.00|> is end-of-text + 107, for English-only and multilingual vocabularies alike."""
    eot = tokens[0] - 1
    timestamp_begin = eot + 107
    is_open, has_text, empty = False, False, 0
    for t in tokens[1:]:
        if t >= timestamp_begin:
            if not is_open:
                is_open, has_text = True, False
            else:
                empty += not has_text
                is_open = False
        elif t == eot:
            empty += is_open and not has_text
            break
        elif t < eot:
            has_text = True
    return empty


# Ground truth, rebuilt exactly as the executor study built the clips.
digit_names = sorted(f"{d}_yweweler_{i}.wav" for d in range(10) for i in range(5))
clips = [{"name": n, "words": [WORDS[int(n[0])]], "kind": "digit"} for n in digit_names]
rng = np.random.default_rng(0)
for s in range(10):
    chosen = rng.choice(len(digit_names), size=4, replace=False)
    clips.append({"name": f"seq_{s}.wav", "words": [WORDS[int(digit_names[c][0])] for c in chosen], "kind": "sequence"})
clips.append(
    {
        "name": "samples_jfk.wav",
        "kind": "long",
        "words": [
            "and",
            "so",
            "my",
            "fellow",
            "americans",
            "ask",
            "not",
            "what",
            "your",
            "country",
            "can",
            "do",
            "for",
            "you",
            "ask",
            "what",
            "you",
            "can",
            "do",
            "for",
            "your",
            "country",
        ],
    }
)

rows = []
for c in clips:
    started = time.time()
    r = subprocess.run(
        [
            binary,
            "--model",
            os.environ.get("WMODEL", "tiny.en"),
            "--input",
            str(clips_dir / c["name"]),
            "--verbose",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    stdout = r.stdout
    spans = re.findall(r"^\s+([\d.]+)s-([\d.]+)s: ?(.*)$", stdout, flags=re.MULTILINE)
    tokens = [list(map(int, t.split(", "))) if t else [] for t in re.findall(r"tokens: \[([\d, ]*)\]", stdout)]
    text = " ".join(t.strip() for _, _, t in spans if t.strip())
    rows.append(
        {
            "clip": c["name"],
            "kind": c["kind"],
            "words": c["words"],
            "returncode": r.returncode,
            "seconds": round(time.time() - started, 2),
            "text": text,
            "spans": [[float(a), float(b), t] for a, b, t in spans],
            "tokens": tokens,
            "skipped_no_speech": stdout.count("no speech detected"),
            "stderr": r.stderr[-400:] if r.returncode else "",
        }
    )
    print(c["name"], repr(text[:70]), tokens[0][:8] if tokens else None, flush=True)

scores = {}
for kind in ("digit", "sequence", "long"):
    sub = [r for r in rows if r["kind"] == kind]
    hyps = [words_of(r["text"]) for r in sub]
    entry = {
        "n": len(sub),
        "empty": sum(1 for r in sub if not r["text"].strip()),
        "wer": float(np.mean([wer(r["words"], h) for r, h in zip(sub, hyps, strict=True)])),
        "clips_with_empty_segment": sum(1 for r in sub if r["tokens"] and empty_segments(r["tokens"][0]) > 0),
    }
    if kind == "digit":
        entry["exact"] = float(
            np.mean(
                [
                    h == r["words"] or [HOMOPHONES.get(w, w) for w in h] == r["words"]
                    for r, h in zip(sub, hyps, strict=True)
                ]
            )
        )
    scores[kind] = entry
report = json.loads(out.read_text()) if out.exists() else {}
report[label] = {"scores": scores, "rows": rows}
out.write_text(json.dumps(report, indent=1))
print(label, json.dumps(scores))
