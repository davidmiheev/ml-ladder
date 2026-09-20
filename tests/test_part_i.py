"""Part I: the front-end conventions and the streaming constraint, checked rather than assumed."""

from __future__ import annotations

import numpy as np
import pytest

from mlladder import audio, vision

torch = pytest.importorskip("torch")


# -- the harness --------------------------------------------------------------


def test_the_speaker_split_shares_no_voice():
    """The leak this part is built to avoid, asserted rather than intended.

    Every speaker says each digit fifty times, so a shuffled split would put
    forty-nine recordings of a voice saying "seven" in training and ask about
    the fiftieth.
    """
    clips = audio.load_fsdd(per_combination=2)
    train, validation, test = audio.split_by_speaker(clips)
    assert set(train.speakers).isdisjoint(validation.speakers)
    assert set(train.speakers).isdisjoint(test.speakers)
    assert set(validation.speakers).isdisjoint(test.speakers)
    assert len(train) + len(validation) + len(test) == len(clips)


def test_pad_batch_reports_the_lengths_it_padded_away():
    waveforms = [np.ones(10, dtype=np.float32), np.ones(4, dtype=np.float32)]
    padded, lengths = audio.pad_batch(waveforms)
    assert padded.shape == (2, 10)
    assert lengths.tolist() == [10, 4]
    assert padded[1, 4:].sum() == 0.0


def test_every_digit_word_is_spellable_in_the_alphabet():
    for word in audio.DIGIT_WORDS:
        assert set(word) <= set(audio.ALPHABET), word


# -- I1: the front end --------------------------------------------------------


def test_hann_window_is_periodic_not_symmetric():
    """The convention that would otherwise look like an FFT bug.

    A periodic Hann divides by N and never reaches zero at the right edge; the
    symmetric form divides by N-1 and does. torchaudio uses periodic, so the
    wrong choice here produces an O(1/N) disagreement in every stage downstream.
    """
    from parts.i_speech.i1_audio_frontend import hann_window

    window = hann_window(8)
    assert window[0] == pytest.approx(0.0)
    assert window[-1] != pytest.approx(0.0)
    assert window[4] == pytest.approx(1.0)


def test_mel_scale_round_trips():
    from parts.i_speech.i1_audio_frontend import hz_to_mel, mel_to_hz

    hertz = np.array([0.0, 100.0, 700.0, 1000.0, 4000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hertz)), hertz, atol=1e-6)


def test_mel_filters_overlap_at_half_height_and_warp_upward():
    """Two structural properties: neighbours share a band, and spacing grows.

    The overlap is what forces the DCT later; the warping is the perceptual
    claim, and both are readable straight off the filterbank.
    """
    from parts.i_speech.i1_audio_frontend import mel_filterbank

    bank = mel_filterbank(n_fft=256, n_mels=40, rate=8000)
    centres = np.asarray([np.argmax(bank[:, m]) for m in range(40)])
    assert (np.diff(centres) >= 0).all()
    spacing = np.diff(np.fft.rfftfreq(256, d=1 / 8000)[centres])
    assert spacing[-5:].mean() > 3 * spacing[:5].mean()
    # Adjacent filters share support; filters two apart mostly do not.
    assert (bank[:, 20] * bank[:, 21]).sum() > 0
    assert (bank[:, 20] * bank[:, 23]).sum() == 0


def test_dct_is_orthonormal():
    from parts.i_speech.i1_audio_frontend import dct_matrix

    basis = dct_matrix(40, 40)
    assert np.allclose(basis @ basis.T, np.eye(40), atol=1e-10)


def test_the_log_guard_convention_changes_the_answer():
    r"""$\log(x + \varepsilon)$ and $\log(\max(x, \varepsilon))$ are not the same guard.

    torchaudio offsets; the obvious reading clamps. On this corpus the quietest
    mel energy is below torchaudio's $10^{-6}$, so the offset acts as a floor on
    silence while a clamp at $10^{-10}$ never fires — and the MFCCs differ by
    four orders of magnitude more than float noise.
    """
    from parts.i_speech.i1_audio_frontend import log_compress

    mel = np.array([[1.0, 1e-3, 1e-8]])
    offset = log_compress(mel, offset=1e-6)
    clamped = log_compress(mel, offset=None, floor=1e-10)
    assert offset[0, 0] == pytest.approx(clamped[0, 0], abs=1e-5)  # loud frames agree
    assert abs(offset[0, 2] - clamped[0, 2]) > 4.0  # quiet ones do not


@pytest.mark.skipif(
    pytest.importorskip("importlib").util.find_spec("torchaudio") is None, reason="torchaudio not installed"
)
def test_the_hand_written_front_end_matches_torchaudio():
    from parts.i_speech.i1_audio_frontend import (
        mel_filterbank,
        mel_spectrogram,
        mfcc,
        power_spectrogram,
        relative_error,
        torchaudio_reference,
    )

    config = audio.FrontEndConfig()
    rng = np.random.default_rng(0)
    waveform = rng.normal(0, 0.1, size=4000).astype(np.float32)
    reference = torchaudio_reference(waveform, config=config, n_mfcc=13)
    assert reference is not None
    bank = mel_filterbank(n_fft=config.n_fft, n_mels=config.n_mels, rate=config.rate, fmax=config.fmax)
    assert relative_error(bank, reference["filterbank"]) < 1e-5
    assert (
        relative_error(power_spectrogram(waveform, n_fft=config.n_fft, hop=config.hop), reference["spectrogram"]) < 1e-5
    )
    assert relative_error(mel_spectrogram(waveform, config=config), reference["mel"]) < 1e-5
    assert relative_error(mfcc(waveform, config=config, n_mfcc=13), reference["mfcc"]) < 1e-4


def test_the_time_frequency_product_is_a_constant():
    """The trade has no free side, and the constant is the proof.

    Halving the window doubles the finest resolvable separation and halves the
    smear of a click, so the product is fixed. Measured across a 16x change in
    window length it should not move at all.
    """
    from parts.i_speech.i1_audio_frontend import impulse_spread, power_spectrogram, tones_resolved, two_tone

    products = []
    for n_fft in (64, 128, 256, 512):
        hop = n_fft // 4
        resolved = [
            separation
            for separation in (20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
            if tones_resolved(
                power_spectrogram(two_tone(8000, separation), n_fft=n_fft, hop=hop).mean(axis=1),
                rate=8000,
                n_fft=n_fft,
                base=1000.0,
                separation=separation,
            )
        ]
        assert resolved, n_fft
        products.append(min(resolved) * impulse_spread(rate=8000, n_fft=n_fft, hop=hop))
    assert max(products) / min(products) < 1.3, products


# -- I3: the vocoder ----------------------------------------------------------


def test_istft_inverts_stft():
    """Overlap-add with window-squared normalisation, not an approximation."""
    from parts.i_speech.i1_audio_frontend import hann_window
    from parts.i_speech.i3_tts_vocoder import _istft, _stft

    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.3, size=2000)
    window = hann_window(256)
    frames = 2000 // 64 + 1
    rebuilt = _istft(_stft(signal, window=window, hop=64, frames=frames), window=window, hop=64)
    length = min(len(signal), len(rebuilt))
    assert np.abs(signal[100 : length - 100] - rebuilt[100 : length - 100]).max() < 1e-8


def test_griffin_lim_improves_with_iterations_and_still_does_not_converge():
    """Phase retrieval is a local optimisation, and this pins both halves of that.

    On a *pure tone* — the easiest possible case, one consistent phase
    assignment — the reconstruction error falls from 0.21 at 16 iterations to
    0.05 at 256 and then stops. It does not reach zero, because alternating
    projection between two non-convex sets converges to a fixed point that need
    not be the global one.

    That plateau is exactly why I3 measures a vocoder ceiling on real audio
    before it measures a synthesiser: a floor this size is inherited by every
    number downstream, and it belongs to Griffin-Lim rather than to the model.
    """
    from parts.i_speech.i1_audio_frontend import hann_window
    from parts.i_speech.i3_tts_vocoder import _stft, griffin_lim

    t = np.arange(4000) / 8000.0
    tone = np.sin(2 * np.pi * 440 * t)
    window = hann_window(256)
    magnitude = np.abs(_stft(tone, window=window, hop=64, frames=4000 // 64 + 1))

    def error(iterations: int) -> float:
        # griffin_lim normalises by the peak, so the absolute level is not
        # recoverable and the comparison is made up to a scale.
        recovered = griffin_lim(magnitude, n_fft=256, hop=64, iterations=iterations, rng=0)
        rebuilt = np.abs(_stft(recovered.astype(np.float64), window=window, hop=64, frames=magnitude.shape[1]))
        scale = np.linalg.norm(magnitude) / max(np.linalg.norm(rebuilt), 1e-12)
        return float(np.linalg.norm(scale * rebuilt - magnitude) / np.linalg.norm(magnitude))

    coarse, fine = error(16), error(256)
    assert fine < coarse / 2, (coarse, fine)  # iterating helps, a lot
    assert fine > 0.01, fine  # and it still does not get there


# -- I4: the streaming constraint ---------------------------------------------


def test_a_causal_model_cannot_see_the_future_and_a_bidirectional_one_can():
    """The structural claim the whole rung rests on, checked without training.

    Perturb the *last* frame of an utterance and look at the *first* frame's
    output. A causal GRU cannot have used it; a bidirectional one must have.
    Nothing about this depends on the weights, which is why it is checked at
    initialisation.
    """
    from parts.i_speech.i2_asr_ctc_whisper import make_speech_crnn
    from parts.i_speech.i4_streaming_latency import make_causal_crnn

    vision.require_torch()
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(0, 1, size=(1, 40, 24)).astype(np.float32))
    disturbed = x.clone()
    disturbed[0, :, -1] += 5.0

    def first_frame_change(factory):
        torch.manual_seed(0)
        model = factory(n_mels=40, num_symbols=16, channels=8, hidden=16)().eval()
        with torch.no_grad():
            return float((model(x)[0, 0] - model(disturbed)[0, 0]).abs().max())

    causal = first_frame_change(make_causal_crnn)
    bidirectional = first_frame_change(make_speech_crnn)
    assert causal < 1e-6, causal
    # Stated as a ratio rather than a threshold: at initialisation the backward
    # pass reaches the first frame through random weights, so its absolute
    # effect is small. The structural difference is not.
    assert bidirectional > 100 * max(causal, 1e-12), (causal, bidirectional)


def test_chunking_with_a_chunk_that_covers_everything_is_the_offline_model():
    from parts.i_speech.i2_asr_ctc_whisper import make_speech_crnn
    from parts.i_speech.i4_streaming_latency import chunked_logits

    vision.require_torch()
    torch.manual_seed(0)
    model = make_speech_crnn(n_mels=40, num_symbols=16, channels=8, hidden=16)().eval()
    rng = np.random.default_rng(1)
    features = rng.normal(0, 1, size=(40, 20)).astype(np.float32)
    streamed = chunked_logits(model, features, chunk=64, lookahead=0, device=torch.device("cpu"))
    with torch.no_grad():
        offline = model(torch.from_numpy(features[None]))[0].numpy()
    assert np.abs(streamed - offline).max() < 1e-5


# -- the rungs run ------------------------------------------------------------


def test_i1_runs():
    from parts.i_speech.i1_audio_frontend import main

    assert main(["--clips", "10", "--per-combination", "1", "--window-sizes", "128", "256"]) == 0


def test_i2_runs():
    from parts.i_speech.i2_asr_ctc_whisper import main

    assert main(["--per-combination", "2", "--epochs", "2", "--eval-every", "1", "--whisper", "none"]) == 0


def test_i3_runs():
    from parts.i_speech.i3_tts_vocoder import main

    assert (
        main(
            [
                "--per-combination",
                "2",
                "--epochs",
                "5",
                "--asr-epochs",
                "2",
                "--griffin-lim-iterations",
                "4",
                "--frames",
                "16",
            ]
        )
        == 0
    )


def test_i4_runs():
    from parts.i_speech.i4_streaming_latency import main

    assert (
        main(
            [
                "--per-combination",
                "2",
                "--epochs",
                "2",
                "--eval-every",
                "1",
                "--chunks",
                "8",
                "32",
                "--lookaheads",
                "0",
                "4",
                "--timing-repeats",
                "2",
            ]
        )
        == 0
    )
