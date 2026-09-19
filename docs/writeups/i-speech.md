# Part I — Speech — recognition & synthesis

> **Status:** not started · TODO(writeup): remove this line when the part is done
> **Numbers:** TODO(writeup): seeds used, e.g. seeds 0-2, reference seed 0 · raw values in [`data/i-metrics.json`](data/i-metrics.json)
> **Reproduce:** `ladder run I1` … , or `uv run python tools/writeup_figures.py I`

*the third modality; reuses C's sequence models and H's CTC on a signal that is continuous in time, where the interesting problems are the front end and the alignment*

TODO(writeup): the through-line — the one idea that ties this part's rungs
together, and how it connects to the parts before it.

---

## I1 — Audio front end from scratch

**Question.** TODO(writeup): what is this rung actually asking?

### The math

TODO(writeup): the objective, its gradient or optimality condition, and the one
property that explains the behaviour you measured. Derive rather than cite;
`$inline$` and `$$display$$` render on GitHub. Say what the math predicts before the results
section says what happened.

**What I built.** TODO(writeup): design decisions and the non-obvious details.

**Results.** TODO(writeup): a table with per-seed ranges, and the figures from
`figures/i/i1_*.png`.

**What surprised me.** TODO(writeup)

**Open threads.** TODO(writeup)

**Reading:** TODO(writeup): the entries from [the reading list](../reading-list.md) that belong here.

---

## I2 — Speech recognition with CTC on a small corpus, and the same audio through Whisper

**Question.** TODO(writeup): what is this rung actually asking?

### The math

TODO(writeup): the objective, its gradient or optimality condition, and the one
property that explains the behaviour you measured. Derive rather than cite;
`$inline$` and `$$display$$` render on GitHub. Say what the math predicts before the results
section says what happened.

**What I built.** TODO(writeup): design decisions and the non-obvious details.

**Results.** TODO(writeup): a table with per-seed ranges, and the figures from
`figures/i/i2_*.png`.

**What surprised me.** TODO(writeup)

**Open threads.** TODO(writeup)

**Reading:** TODO(writeup): the entries from [the reading list](../reading-list.md) that belong here.

---

## I3 — Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening

**Question.** TODO(writeup): what is this rung actually asking?

### The math

TODO(writeup): the objective, its gradient or optimality condition, and the one
property that explains the behaviour you measured. Derive rather than cite;
`$inline$` and `$$display$$` render on GitHub. Say what the math predicts before the results
section says what happened.

**What I built.** TODO(writeup): design decisions and the non-obvious details.

**Results.** TODO(writeup): a table with per-seed ranges, and the figures from
`figures/i/i3_*.png`.

**What surprised me.** TODO(writeup)

**Open threads.** TODO(writeup)

**Reading:** TODO(writeup): the entries from [the reading list](../reading-list.md) that belong here.

---

## I4 — Streaming and latency

**Question.** TODO(writeup): what is this rung actually asking?

### The math

TODO(writeup): the objective, its gradient or optimality condition, and the one
property that explains the behaviour you measured. Derive rather than cite;
`$inline$` and `$$display$$` render on GitHub. Say what the math predicts before the results
section says what happened.

**What I built.** TODO(writeup): design decisions and the non-obvious details.

**Results.** TODO(writeup): a table with per-seed ranges, and the figures from
`figures/i/i4_*.png`.

**What surprised me.** TODO(writeup)

**Open threads.** TODO(writeup)

**Reading:** TODO(writeup): the entries from [the reading list](../reading-list.md) that belong here.

---

## What part I settled

TODO(writeup): numbered takeaways, including the methodological ones.

---

*Figures regenerate with `uv run python tools/writeup_figures.py I`; the
numbers they are quoted against live in [`data/i-metrics.json`](data/i-metrics.json).
See [the writeup conventions](README.md) for the structure this follows.*
