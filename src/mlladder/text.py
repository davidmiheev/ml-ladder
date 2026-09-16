"""Text corpora and character tokenization for part C.

Character level throughout: a byte-pair tokenizer would hide exactly the thing
these rungs are about — how a model carries information across positions — and
a 65-symbol vocabulary keeps the embedding and output layers small enough to
train on CPU.
"""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mlladder import paths
from mlladder.datasets import as_rng

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def tiny_shakespeare(*, limit: int | None = None) -> str:
    """The canonical char-LM corpus (~1.1 MB), cached under ``data/text/``."""
    destination = paths.data_dir() / "text" / "tinyshakespeare.txt"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[text] downloading tiny shakespeare to {destination}")
        request = urllib.request.Request(TINY_SHAKESPEARE_URL, headers={"User-Agent": "ml-ladder"})
        with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    text = destination.read_text(encoding="utf-8")
    return text[:limit] if limit else text


@dataclass
class CharTokenizer:
    """Bijection between the characters present in a corpus and 0..V-1."""

    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(tuple(sorted(set(text))))

    def __post_init__(self) -> None:
        self._to_id = {character: index for index, character in enumerate(self.characters)}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> NDArray[np.int64]:
        return np.array([self._to_id[character] for character in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.characters[int(index)] for index in ids)


def split_corpus(ids: NDArray[np.int64], *, val_fraction: float = 0.1) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Contiguous split — shuffling characters would leak the continuation."""
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def lm_batch(
    ids: NDArray[np.int64], *, batch_size: int, block_size: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """A batch of ``(context, next-character)`` windows.

    Targets are the inputs shifted by one, so position $t$ predicts $t+1$ and a
    single forward pass supplies ``block_size`` training signals per sequence.
    """
    starts = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    inputs = np.stack([ids[start : start + block_size] for start in starts])
    targets = np.stack([ids[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets


# -- the toy translation task for C2 ---------------------------------------

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DATE_INPUT_ALPHABET = "0123456789 /.-,abcdefghijklmnopqrstuvwxyz"

# Filler that surrounds the date when a rung asks for distractor context. The
# words carry no information about the answer, so everything the decoder needs
# still sits in one short span — but a fixed-size context vector now has to
# carry it past all of this.
FILLER_WORDS = (
    "please",
    "confirm",
    "the",
    "appointment",
    "with",
    "our",
    "team",
    "is",
    "scheduled",
    "for",
    "and",
    "will",
    "be",
    "held",
    "at",
    "office",
    "kindly",
    "note",
    "that",
    "we",
    "have",
    "moved",
    "your",
    "booking",
    "to",
    "thanks",
    "regards",
    "as",
    "discussed",
    "earlier",
    "this",
    "week",
)
DATE_OUTPUT_ALPHABET = "0123456789-"


@dataclass(frozen=True)
class DatePair:
    source: str
    target: str


def make_date_pairs(
    n_samples: int, *, context_words: int = 0, rng: int | np.random.Generator | None = None
) -> list[DatePair]:
    """Human-written dates to ISO-8601, in several formats.

    A translation task with an exact answer and a known alignment: every output
    character is a copy of some input span, so an attention map can be checked
    against the truth rather than admired.

    ``context_words`` wraps the date in that many filler words, split randomly
    before and after it. This is the knob that makes the encoder-decoder
    bottleneck bite: padding costs a fixed vector nothing, but real tokens
    between the date and the end of the sentence do.
    """
    generator = as_rng(rng)
    pairs: list[DatePair] = []
    for _ in range(n_samples):
        year = int(generator.integers(1970, 2031))
        month = int(generator.integers(1, 13))
        day = int(generator.integers(1, 29))  # every month has 28 days
        name = MONTHS[month - 1]

        style = int(generator.integers(0, 5))
        if style == 0:
            source = f"{day} {name} {year}"
        elif style == 1:
            source = f"{name} {day}, {year}"
        elif style == 2:
            source = f"{day}/{month}/{year}"
        elif style == 3:
            source = f"{name[:3]} {day} {year}"
        else:
            source = f"{day}.{month}.{year}"

        if context_words > 0:
            before = int(generator.integers(0, context_words + 1))
            words = generator.choice(FILLER_WORDS, size=context_words, replace=True)
            source = " ".join([*words[:before], source, *words[before:]])
        pairs.append(DatePair(source.lower(), f"{year:04d}-{month:02d}-{day:02d}"))
    return pairs


def encode_pairs(
    pairs: list[DatePair], *, source_length: int = 20, target_length: int = 10
) -> tuple[NDArray[np.int64], NDArray[np.int64], CharTokenizer, CharTokenizer]:
    """Pad to fixed lengths and encode both sides.

    Index 0 in each alphabet is the pad symbol, so a mask is just ``ids != 0``.
    """
    source_tokenizer = CharTokenizer(("\x00", *DATE_INPUT_ALPHABET))
    target_tokenizer = CharTokenizer(("\x00", *DATE_OUTPUT_ALPHABET))

    sources = np.zeros((len(pairs), source_length), dtype=np.int64)
    targets = np.zeros((len(pairs), target_length), dtype=np.int64)
    for row, pair in enumerate(pairs):
        source = pair.source[:source_length]
        sources[row, : len(source)] = source_tokenizer.encode(source)
        targets[row, : len(pair.target)] = target_tokenizer.encode(pair.target)
    return sources, targets, source_tokenizer, target_tokenizer


# -- the long-range recall probe for C1 -------------------------------------


def make_recall_task(
    n_samples: int, *, length: int, n_symbols: int = 8, rng: int | np.random.Generator | None = None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Sequences whose label is decided by a single symbol at the *start*.

    ``[cue, noise, noise, ..., noise]`` with the label equal to the cue. Nothing
    in the filler carries information, so a model must route gradient back
    ``length - 1`` steps to learn anything at all — which turns C1's gradient
    reach measurement into an accuracy a cell either gets or does not.

    Returns ``(sequences, labels)`` with symbols ``1..n_symbols`` as cues and 0
    as the filler alphabet's only symbol.
    """
    generator = as_rng(rng)
    cues = generator.integers(1, n_symbols + 1, size=n_samples)
    sequences = np.zeros((n_samples, length), dtype=np.int64)
    sequences[:, 0] = cues
    return sequences, (cues - 1).astype(np.int64)
