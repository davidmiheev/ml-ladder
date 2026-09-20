"""Speech data for part I: a small spoken-digit corpus, loaded without a framework.

Audio is the third modality in this ladder and the first one that is a
*function of time* rather than a grid. That changes what a dataset is: a clip
has no fixed size, the label is a sequence rather than a class, and the split
that matters is by **speaker**, not by clip — a model evaluated on held-out
clips from voices it trained on is answering an easier question than anyone
intends to ask.

The corpus is the Free Spoken Digit Dataset: 3,000 recordings of the digits 0-9
by six speakers, 8 kHz mono. It is small enough to download a slice of on
demand, its filenames are fully determined (``{digit}_{speaker}_{index}.wav``)
so no directory listing is needed, and its transcripts are English words, which
turns digit recognition into a genuine character-sequence task rather than a
ten-way classification.

Reading is via ``scipy.io.wavfile`` rather than soundfile or torchaudio's I/O,
so the data path has no dependency that the rungs are also using as their
reference implementation. Rung I1 compares a hand-written front end against
torchaudio's, and that comparison is worth less if torchaudio also decoded the
file.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mlladder import paths
from mlladder.datasets import as_rng

FSDD_URL = "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/"
FSDD_SPEAKERS = ("george", "jackson", "lucas", "nicolas", "theo", "yweweler")
FSDD_INDICES = 50  # recordings per (digit, speaker)
FSDD_RATE = 8000

#: Transcripts. Spelling the digits out is what makes this a sequence task: a
#: CTC model has to emit "s-e-v-e-n" with the right number of frames per
#: character, where a ten-way classifier would only have to point at a bin.
DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
ALPHABET = tuple(sorted({character for word in DIGIT_WORDS for character in word}))


@dataclass
class Clips:
    """A batch of variable-length recordings with their transcripts.

    Waveforms stay a list rather than a padded array: padding is a modelling
    decision (how much silence, and does the model see a length mask?), and
    baking it into the loader hides it.
    """

    waveforms: list[NDArray[np.float32]]
    digits: NDArray[np.int64]
    speakers: list[str]
    rate: int = FSDD_RATE
    name: str = "fsdd"

    def __len__(self) -> int:
        return len(self.waveforms)

    @property
    def transcripts(self) -> list[str]:
        return [DIGIT_WORDS[digit] for digit in self.digits]

    def durations(self) -> NDArray[np.float32]:
        return np.asarray([len(w) / self.rate for w in self.waveforms], dtype=np.float32)

    def select(self, mask) -> Clips:
        index = np.flatnonzero(np.asarray(mask))
        return Clips(
            waveforms=[self.waveforms[i] for i in index],
            digits=self.digits[index],
            speakers=[self.speakers[i] for i in index],
            rate=self.rate,
            name=self.name,
        )


def _fetch(name: str) -> bytes | None:
    """One recording, cached on disk. ``None`` when the file does not exist."""
    destination = paths.data_dir() / "audio" / "fsdd" / name
    if destination.exists():
        return destination.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(FSDD_URL + name, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    destination.write_bytes(payload)
    return payload


def _decode(payload: bytes) -> NDArray[np.float32]:
    """int16 PCM to float32 in [-1, 1).

    Dividing by 32768 rather than 32767 keeps the mapping exact in binary and
    makes the round trip to int16 lossless, which matters for I3, where a
    synthesised waveform is written out and read back before it is scored.
    """
    from scipy.io import wavfile

    rate, data = wavfile.read(io.BytesIO(payload))
    if rate != FSDD_RATE:  # pragma: no cover - the corpus is uniformly 8 kHz
        raise ValueError(f"expected {FSDD_RATE} Hz, got {rate}")
    if data.dtype != np.int16:  # pragma: no cover
        raise ValueError(f"expected int16 PCM, got {data.dtype}")
    return (data.astype(np.float32) / 32768.0).copy()


def load_fsdd(
    *,
    speakers: tuple[str, ...] = FSDD_SPEAKERS,
    per_combination: int = 20,
    digits: tuple[int, ...] = tuple(range(10)),
    workers: int = 16,
) -> Clips:
    """Download and decode a slice of the corpus, cached under ``data/audio/fsdd``.

    ``per_combination`` recordings of each (digit, speaker) pair, taken from the
    front of the index range so the slice is deterministic and a larger request
    is a superset of a smaller one — which keeps a cache warm across rungs that
    want different amounts.
    """
    names = [
        f"{digit}_{speaker}_{index}.wav"
        for digit in digits
        for speaker in speakers
        for index in range(min(per_combination, FSDD_INDICES))
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = list(pool.map(_fetch, names))

    waveforms: list[NDArray[np.float32]] = []
    labels: list[int] = []
    voices: list[str] = []
    for name, payload in zip(names, payloads, strict=True):
        if payload is None:  # pragma: no cover - the index range is known good
            continue
        digit, speaker, _ = name.removesuffix(".wav").split("_")
        waveforms.append(_decode(payload))
        labels.append(int(digit))
        voices.append(speaker)
    return Clips(waveforms=waveforms, digits=np.asarray(labels, dtype=np.int64), speakers=voices)


def split_by_speaker(clips: Clips, *, validation: str = "theo", test: str = "yweweler") -> tuple[Clips, Clips, Clips]:
    """Speaker-independent split: no voice appears on both sides.

    The default alternative — shuffling clips — leaks. Each speaker contributes
    fifty recordings of every digit, so a random split puts forty-nine examples
    of a voice saying "seven" in training and asks about the fiftieth. That
    measures how well the model memorised a voice, and the number it produces is
    much better than the one anybody wants.
    """
    voices = np.asarray(clips.speakers)
    return (
        clips.select(~np.isin(voices, [validation, test])),
        clips.select(voices == validation),
        clips.select(voices == test),
    )


def pad_batch(waveforms: list[NDArray[np.float32]]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Right-pad to the longest clip, returning the true lengths alongside.

    The lengths are not optional bookkeeping. A CTC loss needs the unpadded
    input length per item, and a model that pools over padding learns that
    silence is evidence.
    """
    lengths = np.asarray([len(w) for w in waveforms], dtype=np.int64)
    padded = np.zeros((len(waveforms), int(lengths.max())), dtype=np.float32)
    for row, waveform in enumerate(waveforms):
        padded[row, : len(waveform)] = waveform
    return padded, lengths


def resample_poly(waveform: NDArray[np.float32], source: int, target: int) -> NDArray[np.float32]:
    """Rational resampling via ``scipy.signal.resample_poly``.

    Needed because the corpus is 8 kHz and the pretrained baseline in I2 expects
    16 kHz. Upsampling invents no information — everything above 4 kHz is
    interpolation — which is worth saying out loud, since a model trained on
    wideband speech is being handed a band-limited signal and its errors are
    partly about that.
    """
    from math import gcd

    from scipy.signal import resample_poly as _resample

    divisor = gcd(source, target)
    return _resample(waveform, target // divisor, source // divisor).astype(np.float32)


def speaker_balanced_subset(clips: Clips, *, per_digit: int, rng=None) -> Clips:
    """``per_digit`` clips of each digit, drawn across whatever speakers are present."""
    generator = as_rng(rng)
    keep: list[int] = []
    for digit in range(10):
        members = np.flatnonzero(clips.digits == digit)
        generator.shuffle(members)
        keep.extend(members[:per_digit].tolist())
    mask = np.zeros(len(clips), dtype=bool)
    mask[sorted(keep)] = True
    return clips.select(mask)


@dataclass
class FrontEndConfig:
    """The parameters every rung in part I shares, in one place.

    Written down as a dataclass because these are not tuning knobs to be varied
    per rung: a mel spectrogram computed with a different hop in I2 than in I3
    would make the two rungs' numbers incomparable, and the point of I3's
    round-trip metric is that they are comparable.
    """

    rate: int = FSDD_RATE
    n_fft: int = 256  # 32 ms at 8 kHz
    hop: int = 64  # 8 ms
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float = field(default=FSDD_RATE / 2)

    @property
    def window_ms(self) -> float:
        return 1000.0 * self.n_fft / self.rate

    @property
    def hop_ms(self) -> float:
        return 1000.0 * self.hop / self.rate

    def frames(self, samples: int) -> int:
        """Frame count for a centred STFT, matching torch.stft(center=True)."""
        return samples // self.hop + 1
