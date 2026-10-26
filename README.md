# Interview & Voice Analysis

Final code for Korean interview transcription, manual review, language analysis
and acoustic comparison of voice samples.

[한국어](README.ko.md) · [Setup and commands](docs/RUN_GUIDE_EN.txt)

## Three workflows

| Task | Code folder | What it does | Outputs |
| --- | --- | --- | --- |
| **1. Prepare and review transcripts** | [transcript_review](pipeline/transcript_review) | Extract audio, transcribe, align word times, replay audio or linked video, mark BAD/overlap, screen timing | Reviewed and timing-qualified word JSON |
| **2. Analyze interviews** | [interview_analysis](pipeline/interview_analysis) | librosa acoustic features, KOTE text emotion, rule-based intent, participant BAR summaries | Per-clip JSON and summary CSVs |
| **3. Compare voice samples** | [voice_samples](pipeline/voice_samples) | Apply the same acoustic method to supplied-script voices; compare clips/voices, wording checks and timing sensitivity | Per-clip JSON and voice comparison CSVs |

Use **`run.py`** for all three workflows. Files in `_core` are internal calculation
helpers; they are not extra steps to run. Voice analysis reuses the interview
acoustic implementation.

```console
python run.py review --help
python run.py interview --help
python run.py voices --help
```

The [run guide](docs/RUN_GUIDE_EN.txt) gives the complete sequence, dependencies,
model paths, input layout and output definitions. Python 3.12 was used; ASR and
analysis use separate environments.

## Transcript review

```console
python run.py review listen --project project
```

On Windows: **Enter** saves and plays the next answer; **p** replays;
**4b** marks word 4 as bad text; **4o** marks overlap; **B** marks the whole
transcript unreliable; **v** opens a linked video; **q** exits.
Type each command and press Enter. Optional video links use `project/video_map.csv`.
Audio/video playback belongs to this review workflow.

Existing word-level JSON can enter at validation/review. Newly received recordings
first require audio preparation, transcription and alignment. Clip boundaries
and question IDs are supplied inputs; the code does not identify them automatically.

## Supporting files

- `config/`: recorded model metadata, measurement settings and fixed BAR reference.
- `requirements/`: recorded package versions; see the run guide for installation limits.
- `tests/`, `example/`: method checks and a fictional JSON/tone fixture.

```console
python -m unittest discover -s tests -v
python example/make_demo.py
python run.py voices analyze --project example/project --out demo_output
python -m example.check_result --result demo_output
```

The acoustic example requires the analysis environment. It checks numerical
processing using a generated tone, not speech-recognition accuracy.

## Measurement notes

- KOTE classifies **text emotion**. Intent scores are rule evidence, not probabilities.
- F0 and RMS clip summaries average word measurements. Speech rate uses summed
  word durations, not total elapsed clip duration.
- BAR provides auxiliary behavioural indicators, not validated personality
  diagnoses or the human-assigned Group A/B labels.
- Compare voice candidates within the same group and question; A and B use
  different scripts.

This package contains code and settings. Recordings, real transcripts, participant
tables, individual analysis results and model weights are supplied separately.
