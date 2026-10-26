# Language, Voice & Interview JSON Analysis

Final research code for Korean interview transcription, word-level review,
text analysis and acoustic voice analysis.

[한국어 안내](README.ko.md) · [Run guide](docs/RUN_GUIDE_EN.txt) · [First-time GitHub guide](docs/FIRST_GITHUB_KO.md)

## Analysis stages

| Stage | Method | Entry point |
| --- | --- | --- |
| Audio preparation | FFmpeg, 16-kHz mono PCM16; supplied clip boundaries | `code/prepare_audio.py` |
| Transcription | Local faster-whisper large-v3-turbo, Korean | `code/transcribe.py` |
| Word alignment | WhisperX and Korean wav2vec2 alignment model | `code/align.py` |
| Transcript review | Windows terminal; replay, bad-text/overlap flags | `code/review.py` |
| Timing quality | Duration and Silero VAD speech-coverage screening | `code/timing_qc.py` |
| Text emotion / intent | KOTE 44 labels and Korean intent rules | `code/analyze.py` |
| Voice acoustics | librosa pYIN F0, RMS, word-duration speech rate and QC | `code/analyze.py` |
| Interview indicators | BAR aggregation with the fixed study reference | `code/bar_profile.py` |
| Voice comparison | Per-clip/voice summaries and timing sensitivity | `code/compare_voices.py` |
| JSON validation | Fields, finite timestamps and interval checks | `code/validate_json.py` |

## Start here

Read the [run guide](docs/RUN_GUIDE_EN.txt) for environment setup, model paths,
input layouts and separate interview / supplied-script workflows.
Python 3.12 was used. Analysis and ASR use separate environments.

The method and input tests require only Python:

```console
python -m unittest discover -s tests -v
```

An optional acoustic smoke test uses a locally generated tone, with **no recorded
speech or participant data**. Install the analysis dependencies first:

```console
python example/make_demo.py
python code/analyze.py --project example/project --out demo_output --stages prosody
python code/check_example.py --result demo_output
```

The tone and its fictional word intervals test numerical processing only.
They do not test transcription, speech recognition accuracy or vocal realism.

## Repository layout

```text
code/          Analysis entry points and calculation helpers
config/        Model metadata, measurement settings and fixed BAR reference
requirements/  Recorded dependency versions
tests/         Method and JSON-input checks
example/       Fictional JSON fixture and deterministic tone generator
docs/          Detailed run guide and GitHub instructions
```

## Interpretation

- KOTE classifies text emotion; it is not vocal emotion recognition.
- Intent scores are rule evidence, not probabilities.
- F0 and RMS clip summaries average word-level measurements. Speech rate uses
  summed word durations rather than the full elapsed clip duration.
- BAR scores are auxiliary behavioural indicators, not validated personality
  diagnoses or the human-assigned Group A/B labels.
- Compare synthetic voice candidates within the same group and question;
  Group A and Group B use different scripts.

Recordings, real transcripts, participant tables, analysis datasets, credentials
and model weights are kept outside this repository. Pre-existing interview JSON
is ingested as supplied. The local ASR step is used for newly provided recordings.
