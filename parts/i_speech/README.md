# I. Speech — recognition & synthesis

*the third modality; reuses C's sequence models and H's CTC on a signal that is continuous in time, where the interesting problems are the front end and the alignment*


**Lab:** [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| I1 | Audio front end from scratch — STFT, mel filterbank and MFCCs, against torchaudio's | `i1_audio_frontend.py` |
| I2 | Speech recognition with CTC on a small corpus, and the same audio through Whisper | `i2_asr_ctc_whisper.py` |
| I3 | Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening | `i3_tts_vocoder.py` |
| I4 | Streaming and latency — chunked inference against full-utterance, with the accuracy it costs | `i4_streaming_latency.py` |

Run any rung with `ladder run I1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
