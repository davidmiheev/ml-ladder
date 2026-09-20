r"""I1. Audio front end from scratch — STFT, mel filterbank and MFCCs, against torchaudio's

Lab reference: [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

A speech model does not see a waveform. It sees whatever the front end produced,
and the front end is four decisions stacked on top of each other — frame length,
window, frequency warping, and whether to decorrelate. Every one of them throws
information away on purpose, and the choices are old enough that they are
usually copied rather than made.

This rung writes the stack out and then measures the three claims inside it that
are actually checkable:

1. **The window length is a trade, not a setting.** Framing buys time resolution
   with frequency resolution at a fixed exchange rate. Two tones $\Delta f$ apart
   are resolvable only when the window is long enough to hold a full beat period,
   $\Delta f \gtrsim f_s/N$, and the same window smears an impulse across
   $N$ samples. The product is bounded below — a discrete Gabor limit — and a
   sweep over $N$ should trace it out.
2. **The mel scale is a claim about hearing, not about signals.** Nothing in the
   physics says 1 kHz deserves more filters than 3 kHz. The claim is that human
   frequency discrimination is roughly logarithmic above ~700 Hz, and the
   filterbank encodes it as a warping whose asymmetry can simply be counted.
3. **The DCT is there to decorrelate.** Overlapping triangular filters guarantee
   that adjacent mel bands are highly correlated; MFCCs exist because a
   diagonal-covariance model cannot cope with that. The decorrelation is a
   number, and it is the reason the last step exists at all.

What is *not* written from scratch is the FFT, which is a numerical-methods
exercise rather than a speech one; `numpy.fft.rfft` does that step. Everything
around it — framing, windowing, the power spectrum, the mel warping, the
filterbank, the log, the DCT — is here, and each stage is diffed against
torchaudio separately so a disagreement lands on the stage that caused it.

Run with ``ladder run I1`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, audio, experiment, plotting


def hann_window(length: int) -> NDArray[np.float64]:
    r"""$w[n] = \tfrac{1}{2}\left(1 - \cos\frac{2\pi n}{N}\right)$, periodic.

    Periodic (divide by $N$) rather than symmetric (divide by $N-1$), because
    the window is used for spectral analysis of a signal assumed to repeat, and
    the periodic form is the one that sums to a constant under overlap-add.
    torchaudio's default is also periodic; using the symmetric form here would
    produce a disagreement of order $1/N$ that looks like a bug in the FFT.
    """
    n = np.arange(length, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / length))


def frame(waveform: NDArray[np.float32], *, n_fft: int, hop: int, center: bool = True) -> NDArray[np.float64]:
    """Overlapping frames, one per row.

    ``center=True`` pads by ``n_fft // 2`` at both ends with a *reflection*, so
    frame $t$ is centred on sample $t \\cdot \\text{hop}$ and the first frame is
    not half silence. Reflection rather than zeros because zero-padding
    introduces a discontinuity at the boundary, and a discontinuity is broadband
    — it puts energy at every frequency in the first and last frames.
    """
    signal = np.asarray(waveform, dtype=np.float64)
    if center:
        signal = np.pad(signal, n_fft // 2, mode="reflect")
    count = 1 + (len(signal) - n_fft) // hop
    indices = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return signal[indices]


def power_spectrogram(waveform, *, n_fft: int, hop: int) -> NDArray[np.float64]:
    """$|X[t, k]|^2$ — frames, windowed, real FFT, squared magnitude."""
    frames = frame(waveform, n_fft=n_fft, hop=hop)
    spectrum = np.fft.rfft(frames * hann_window(n_fft)[None, :], axis=1)
    return (spectrum.real**2 + spectrum.imag**2).T  # (freq, time), like torchaudio


def hz_to_mel(hertz):
    r"""HTK: $m = 2595 \log_{10}(1 + f/700)$.

    One of two conventions in common use — the other, Slaney's, is linear below
    1 kHz and logarithmic above, with a different constant. They disagree by
    enough to move every filter edge, so "the mel scale" is not a well-defined
    phrase without saying which. torchaudio defaults to HTK and so does this.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(hertz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(*, n_fft: int, n_mels: int, rate: int, fmin: float = 0.0, fmax: float | None = None):
    r"""Triangular filters, evenly spaced on the mel scale.

    $n_{\text{mels}} + 2$ points are spaced uniformly in mel and mapped back to
    hertz; each filter rises linearly from the point below it to its centre and
    falls to the point above, so adjacent filters overlap at half height and
    **every pair of neighbours shares a band**. That overlap is what the DCT
    later has to undo.

    No area normalisation, matching torchaudio's ``norm=None``. With Slaney
    normalisation each filter is scaled by $2/(f_{\text{high}} - f_{\text{low}})$,
    which changes every coefficient's magnitude and therefore any comparison
    that does not say which was used.
    """
    fmax = rate / 2 if fmax is None else fmax
    bins = np.fft.rfftfreq(n_fft, d=1.0 / rate)
    points = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    filters = np.zeros((len(bins), n_mels), dtype=np.float64)
    for m in range(n_mels):
        low, centre, high = points[m], points[m + 1], points[m + 2]
        rising = (bins - low) / max(centre - low, 1e-12)
        falling = (high - bins) / max(high - centre, 1e-12)
        filters[:, m] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def mel_spectrogram(waveform, *, config: audio.FrontEndConfig) -> NDArray[np.float64]:
    power = power_spectrogram(waveform, n_fft=config.n_fft, hop=config.hop)
    filters = mel_filterbank(
        n_fft=config.n_fft, n_mels=config.n_mels, rate=config.rate, fmin=config.fmin, fmax=config.fmax
    )
    return filters.T @ power


def dct_matrix(n_out: int, n_in: int) -> NDArray[np.float64]:
    r"""Orthonormal DCT-II, $n_{\text{out}} \times n_{\text{in}}$.

    $$C[k, n] = \sqrt{\tfrac{2}{N}}\,\cos\!\left(\tfrac{\pi k (2n + 1)}{2N}\right),
    \qquad C[0, n] \text{ scaled by } \tfrac{1}{\sqrt{2}} .$$

    Orthonormal so the transform preserves energy and can be inverted by its own
    transpose — which is what makes truncating to 13 coefficients a *projection*
    with a measurable residual rather than an arbitrary crop.
    """
    n = np.arange(n_in, dtype=np.float64)
    k = np.arange(n_out, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    basis[0] /= np.sqrt(2.0)
    return basis


def log_compress(mel, *, offset: float | None = 1e-6, floor: float = 1e-10):
    r"""$\log$ of a mel spectrogram, by one of the two conventions in use.

    The log is not cosmetic: it turns the multiplicative relationship between a
    vocal-tract filter and a glottal source into an additive one, which is the
    only reason a linear transform afterwards can separate them — the whole
    cepstral idea in one line. But $\log 0$ is undefined and every
    implementation therefore picks a guard, and there are two:

    - **offset**, $\log(x + \varepsilon)$ — torchaudio's, with
      $\varepsilon = 10^{-6}$. Smooth, and it *shifts every value*, not just the
      small ones.
    - **clamp**, $\log(\max(x, \varepsilon))$ — the obvious reading, and the
      one this rung wrote first. It changes nothing above the floor.

    They are not interchangeable and the difference is not in the noise. On this
    corpus the quietest mel energy is $3.9\times10^{-7}$, *below* torchaudio's
    offset, so the offset is acting as a floor on silence while the clamp at
    $10^{-10}$ never fires at all. The rung measures what that costs.
    """
    if offset is not None:
        return np.log(np.asarray(mel, dtype=np.float64) + offset)
    return np.log(np.maximum(np.asarray(mel, dtype=np.float64), floor))


def mfcc(waveform, *, config: audio.FrontEndConfig, n_mfcc: int = 13, offset: float | None = 1e-6):
    """Log-mel, then DCT-II, truncated to ``n_mfcc``."""
    log_mel = log_compress(mel_spectrogram(waveform, config=config), offset=offset)
    return dct_matrix(n_mfcc, config.n_mels) @ log_mel


# -- the measurements ---------------------------------------------------------


def two_tone(rate: int, separation: float, *, base: float = 1000.0, duration: float = 0.25):
    """Two equal-amplitude sinusoids ``separation`` Hz apart."""
    t = np.arange(int(rate * duration), dtype=np.float64) / rate
    return (np.sin(2 * np.pi * base * t) + np.sin(2 * np.pi * (base + separation) * t)).astype(np.float32)


def tones_resolved(spectrum: NDArray[np.float64], *, rate: int, n_fft: int, base: float, separation: float) -> bool:
    """Are the two tones two peaks, or one?

    The textbook criterion: a dip between the peaks. Taken on the averaged
    magnitude spectrum, and required to fall below 70% of the smaller peak so a
    ripple does not count as resolution.
    """
    bins = np.fft.rfftfreq(n_fft, d=1.0 / rate)
    band = (bins >= base - 4 * separation) & (bins <= base + 5 * separation)
    profile = spectrum[band]
    if len(profile) < 3:
        return False
    interior = profile[1:-1]
    peaks = [i + 1 for i in range(len(interior)) if profile[i + 1] > profile[i] and profile[i + 1] > profile[i + 2]]
    if len(peaks) < 2:
        return False
    first, last = peaks[0], peaks[-1]
    return bool(profile[first : last + 1].min() < 0.7 * min(profile[first], profile[last]))


def impulse_spread(*, rate: int, n_fft: int, hop: int) -> float:
    """Milliseconds over which a single-sample click is smeared.

    The other half of the trade, measured on the same axis as the frequency
    criterion: count the frames carrying at least 1% of the peak frame's energy.
    """
    click = np.zeros(4096, dtype=np.float32)
    click[2048] = 1.0
    energy = power_spectrogram(click, n_fft=n_fft, hop=hop).sum(axis=0)
    return float((energy > 0.01 * energy.max()).sum() * hop / rate * 1000.0)


def mean_offdiagonal_correlation(features: NDArray[np.float64]) -> float:
    """Mean |correlation| between distinct feature dimensions over time."""
    matrix = np.corrcoef(features)
    matrix = np.nan_to_num(matrix)
    size = matrix.shape[0]
    off = ~np.eye(size, dtype=bool)
    return float(np.abs(matrix[off]).mean())


def torchaudio_reference(waveform, *, config: audio.FrontEndConfig, n_mfcc: int):
    """The same four stages from torchaudio, or ``None`` if it is not installed.

    Configured to match this module's conventions explicitly — centred frames,
    periodic Hann, HTK mel, no filter normalisation, power spectrogram — because
    every one of those is a default that could differ and each would show up as
    a different-looking mismatch.
    """
    try:
        import torch
        import torchaudio.functional as taf
        import torchaudio.transforms as tat
    except ImportError:  # pragma: no cover - optional dependency
        return None

    signal = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    spectrogram = tat.Spectrogram(
        n_fft=config.n_fft, hop_length=config.hop, power=2.0, center=True, pad_mode="reflect"
    )(signal)
    filters = taf.melscale_fbanks(
        config.n_fft // 2 + 1, config.fmin, config.fmax, config.n_mels, config.rate, norm=None, mel_scale="htk"
    )
    mel = tat.MelSpectrogram(
        sample_rate=config.rate,
        n_fft=config.n_fft,
        hop_length=config.hop,
        n_mels=config.n_mels,
        f_min=config.fmin,
        f_max=config.fmax,
        power=2.0,
        norm=None,
        mel_scale="htk",
        center=True,
    )(signal)
    coefficients = tat.MFCC(
        sample_rate=config.rate,
        n_mfcc=n_mfcc,
        log_mels=True,
        melkwargs=dict(
            n_fft=config.n_fft,
            hop_length=config.hop,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax,
            power=2.0,
            norm=None,
            mel_scale="htk",
            center=True,
        ),
    )(signal)
    return {
        "spectrogram": spectrogram.numpy().astype(np.float64),
        "filterbank": filters.numpy().astype(np.float64),
        "mel": mel.numpy().astype(np.float64),
        "mfcc": coefficients.numpy().astype(np.float64),
    }


def relative_error(mine: NDArray[np.float64], theirs: NDArray[np.float64]) -> float:
    difference = np.asarray(mine) - np.asarray(theirs)
    return float(np.linalg.norm(difference) / max(np.linalg.norm(theirs), 1e-12))


@experiment(
    "i1_audio_frontend",
    defaults={
        "clips": 40,
        "per_combination": 4,
        "n_mfcc": 13,
        # The window sweep: 4 ms to 128 ms at 8 kHz.
        "window_sizes": [32, 64, 128, 256, 512, 1024],
        "tone_separations": [20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
        "tone_base": 1000.0,
    },
    description="STFT, mel filterbank and MFCCs written from scratch and diffed against torchaudio.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Audio front end from scratch — STFT, mel filterbank and MFCCs, against torchaudio's"""
    config = audio.FrontEndConfig()
    summary: dict[str, float] = {}
    print(f"[i1] {config.rate} Hz, window {config.window_ms:.0f} ms, hop {config.hop_ms:.0f} ms, {config.n_mels} mels")

    clips = audio.load_fsdd(per_combination=exp["per_combination"])
    clips = audio.speaker_balanced_subset(clips, per_digit=max(1, exp["clips"] // 10), rng=exp.seed)
    summary["corpus/clips"] = float(len(clips))
    summary["corpus/mean_duration_s"] = float(clips.durations().mean())
    print(f"[i1] {len(clips)} clips, mean {clips.durations().mean() * 1000:.0f} ms")

    # -- 1. does the hand-written stack agree with the reference? -------------
    errors: dict[str, list[float]] = {"spectrogram": [], "mel": [], "mfcc": [], "mfcc_clamped": []}
    reference_available = False
    for waveform in clips.waveforms:
        reference = torchaudio_reference(waveform, config=config, n_mfcc=exp["n_mfcc"])
        if reference is None:
            break
        reference_available = True
        errors["spectrogram"].append(
            relative_error(power_spectrogram(waveform, n_fft=config.n_fft, hop=config.hop), reference["spectrogram"])
        )
        errors["mel"].append(relative_error(mel_spectrogram(waveform, config=config), reference["mel"]))
        errors["mfcc"].append(relative_error(mfcc(waveform, config=config, n_mfcc=exp["n_mfcc"]), reference["mfcc"]))
        errors["mfcc_clamped"].append(
            relative_error(mfcc(waveform, config=config, n_mfcc=exp["n_mfcc"], offset=None), reference["mfcc"])
        )
    summary["reference/available"] = float(reference_available)
    if reference_available:
        bank = mel_filterbank(
            n_fft=config.n_fft, n_mels=config.n_mels, rate=config.rate, fmin=config.fmin, fmax=config.fmax
        )
        reference = torchaudio_reference(clips.waveforms[0], config=config, n_mfcc=exp["n_mfcc"])
        summary["reference/filterbank"] = relative_error(bank, reference["filterbank"])
        for stage, values in errors.items():
            summary[f"reference/{stage}"] = float(np.max(values))
        summary["reference/log_guard_cost"] = summary["reference/mfcc_clamped"] / max(summary["reference/mfcc"], 1e-12)
        quietest = float(min(mel_spectrogram(w, config=config).min() for w in clips.waveforms))
        summary["mel/quietest_energy"] = quietest
        print(
            f"[i1] vs torchaudio (worst of {len(errors['mel'])} clips): filterbank "
            f"{summary['reference/filterbank']:.2e}, spectrogram {summary['reference/spectrogram']:.2e}, "
            f"mel {summary['reference/mel']:.2e}, mfcc {summary['reference/mfcc']:.2e}"
        )
        print(
            f"[i1] the same MFCCs with a clamp instead of an offset: {summary['reference/mfcc_clamped']:.2e}, "
            f"{summary['reference/log_guard_cost']:.0f}x worse. The quietest mel energy here is {quietest:.1e}, "
            f"below torchaudio's 1e-06 offset -- so the guard is not guarding, it is shifting"
        )
    else:  # pragma: no cover - optional dependency
        print("[i1] torchaudio is not installed; the reference comparison is skipped, not assumed")

    # -- 2. the time-frequency trade -----------------------------------------
    resolution_limit: list[float] = []
    spreads: list[float] = []
    for n_fft in exp["window_sizes"]:
        hop = max(1, n_fft // 4)
        resolved = [
            separation
            for separation in exp["tone_separations"]
            if tones_resolved(
                power_spectrogram(two_tone(config.rate, separation, base=exp["tone_base"]), n_fft=n_fft, hop=hop).mean(
                    axis=1
                ),
                rate=config.rate,
                n_fft=n_fft,
                base=exp["tone_base"],
                separation=separation,
            )
        ]
        finest = min(resolved) if resolved else float("nan")
        spread = impulse_spread(rate=config.rate, n_fft=n_fft, hop=hop)
        resolution_limit.append(finest)
        spreads.append(spread)
        summary[f"window/{n_fft}/finest_separation_hz"] = finest
        summary[f"window/{n_fft}/impulse_spread_ms"] = spread
        summary[f"window/{n_fft}/bin_width_hz"] = config.rate / n_fft
        print(
            f"[i1] window {n_fft:>5} ({1000 * n_fft / config.rate:6.1f} ms): "
            f"resolves {finest:>5.0f} Hz, smears a click over {spread:5.1f} ms"
        )
    finite = [(f, s) for f, s in zip(resolution_limit, spreads, strict=True) if np.isfinite(f)]
    if len(finite) >= 2:
        products = [f * s for f, s in finite]
        summary["window/resolution_product_mean"] = float(np.mean(products))
        summary["window/resolution_product_spread"] = float(max(products) / max(min(products), 1e-12))
        print(
            f"[i1] the product (Hz x ms) stays within {summary['window/resolution_product_spread']:.2f}x "
            f"across a {max(exp['window_sizes']) // min(exp['window_sizes'])}x change in window length "
            f"-- that is the trade, and it does not have a free side"
        )

    # -- 3. what the mel warping actually does -------------------------------
    bank = mel_filterbank(
        n_fft=config.n_fft, n_mels=config.n_mels, rate=config.rate, fmin=config.fmin, fmax=config.fmax
    )
    bins = np.fft.rfftfreq(config.n_fft, d=1.0 / config.rate)
    centres = np.asarray([bins[np.argmax(bank[:, m])] for m in range(config.n_mels)])
    half = config.rate / 4  # 2 kHz at 8 kHz sampling
    below = int((centres < half).sum())
    summary["mel/filters_below_half_nyquist"] = float(below)
    summary["mel/filters_above_half_nyquist"] = float(config.n_mels - below)
    summary["mel/lowest_spacing_hz"] = float(np.diff(centres)[:5].mean())
    summary["mel/highest_spacing_hz"] = float(np.diff(centres)[-5:].mean())
    summary["mel/spacing_ratio"] = summary["mel/highest_spacing_hz"] / max(summary["mel/lowest_spacing_hz"], 1e-9)
    print(
        f"[i1] mel warping: {below} of {config.n_mels} filters sit below {half:.0f} Hz, the lower half of the "
        f"band; spacing grows from {summary['mel/lowest_spacing_hz']:.0f} Hz to "
        f"{summary['mel/highest_spacing_hz']:.0f} Hz, a factor of {summary['mel/spacing_ratio']:.1f}"
    )

    # -- 4. the DCT's job ----------------------------------------------------
    log_mels = np.concatenate([log_compress(mel_spectrogram(w, config=config)) for w in clips.waveforms], axis=1)
    coefficients = dct_matrix(exp["n_mfcc"], config.n_mels) @ log_mels
    summary["dct/correlation_before"] = mean_offdiagonal_correlation(log_mels)
    summary["dct/correlation_after"] = mean_offdiagonal_correlation(coefficients)
    summary["dct/decorrelation"] = summary["dct/correlation_before"] / max(summary["dct/correlation_after"], 1e-12)
    # Truncation is a projection, so its cost is measurable rather than assumed.
    basis = dct_matrix(config.n_mels, config.n_mels)
    kept = basis[: exp["n_mfcc"]].T @ (basis[: exp["n_mfcc"]] @ log_mels)
    summary["dct/truncation_residual"] = float(np.linalg.norm(log_mels - kept) / max(np.linalg.norm(log_mels), 1e-12))
    print(
        f"[i1] mel bands correlate at {summary['dct/correlation_before']:.3f} off-diagonal; after the DCT "
        f"{summary['dct/correlation_after']:.3f}, a {summary['dct/decorrelation']:.1f}x drop. Keeping "
        f"{exp['n_mfcc']} of {config.n_mels} coefficients loses "
        f"{summary['dct/truncation_residual'] * 100:.1f}% of the log-mel energy"
    )

    # -- figures --------------------------------------------------------------
    example = clips.waveforms[0]
    exp.log_figure(
        "pipeline",
        plotting.plot_architecture(
            {
                "The four decisions": [
                    ("waveform", f"{config.rate} Hz, ~{clips.durations().mean() * 1000:.0f} ms"),
                    (
                        "FRAME + WINDOW",
                        f"{config.n_fft} samples = {config.window_ms:.0f} ms, hop {config.hop_ms:.0f} ms",
                    ),
                    ("|rFFT|^2", f"{config.n_fft // 2 + 1} bins, {config.rate / config.n_fft:.1f} Hz apart"),
                    ("MEL FILTERBANK", f"{config.n_mels} triangles, spacing x{summary['mel/spacing_ratio']:.1f}"),
                    ("log", "vocal tract and source become additive"),
                    ("DCT-II, keep 13", "decorrelate, then truncate"),
                ],
            },
            highlight={"The four decisions": [1, 3, 5]},
            title="Each highlighted step throws information away on purpose",
        ),
    )
    exp.log_figure(
        "filterbank",
        plotting.plot_curves(
            {f"filter {m}": bank[:, m].tolist() for m in range(0, config.n_mels, 4)},
            x=bins.tolist(),
            title="Every fourth mel filter -- narrow and dense low, wide and sparse high",
            xlabel="frequency (Hz)",
            ylabel="weight",
        ),
    )
    exp.log_figure(
        "stages",
        plotting.plot_spectrograms(
            {
                "log |X|^2": np.log(np.maximum(power_spectrogram(example, n_fft=config.n_fft, hop=config.hop), 1e-10)),
                "log mel": log_compress(mel_spectrogram(example, config=config)),
                "MFCC": mfcc(example, config=config, n_mfcc=exp["n_mfcc"]),
            },
            title=f'one clip of "{clips.transcripts[0]}" through the stack',
            rate=config.rate,
            hop=config.hop,
        ),
    )
    exp.log_figure(
        "resolution",
        plotting.plot_curves(
            {
                "finest separation resolved (Hz)": resolution_limit,
                "impulse spread (ms)": spreads,
            },
            x=exp["window_sizes"],
            title="The trade: longer windows buy frequency with time",
            xlabel="window (samples)",
            ylabel="Hz, or ms",
            logx=True,
            logy=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "decorrelation",
        plotting.plot_comparison(
            {"log-mel bands": summary["dct/correlation_before"], "MFCCs": summary["dct/correlation_after"]},
            title="Mean off-diagonal |correlation| -- what the DCT is for",
            ylabel="|correlation|",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
