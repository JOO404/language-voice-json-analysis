"""Local Korean ASR for project/data/wav/<id>/<clip>.wav.

interview: fresh per-clip transcription; voice-check: independent script check.
"""
import argparse
import dataclasses
import difflib
import importlib.metadata
import os
from pathlib import Path
import re
import site
from analyze import read, write


def norm(text):
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New folder for ASR JSON')
    parser.add_argument('--model', type=Path, required=True, help='Local CTranslate2 large-v3-turbo directory')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--mode', choices=['interview','voice-check'], default='interview')
    parser.add_argument('--scripts', type=Path, help='Folder of supplied text JSON (voice-check)')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output must be new or empty')
    if not args.model.is_dir():
        parser.error('Local model directory not found')
    if args.mode == 'voice-check' and not args.scripts:
        parser.error('voice-check requires --scripts')
    wav_root = args.project/'data/wav'
    files = sorted(wav_root.rglob('*.wav'))
    if not files:
        parser.error('No WAV files found')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    handles = []
    if os.name == 'nt' and args.device == 'cuda':
        for folder in site.getsitepackages():
            for dll_dir in sorted({p.parent for p in (Path(folder)/'nvidia').rglob('*.dll')}):
                handles.append(os.add_dll_directory(str(dll_dir)))
                os.environ['PATH'] = str(dll_dir)+os.pathsep+os.environ.get('PATH','')
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model.resolve()), device=args.device,
        compute_type='int8_float16' if args.device == 'cuda' else 'int8',
        cpu_threads=6, local_files_only=True)
    settings = dict(language='ko', task='transcribe', beam_size=5,
        temperature=[0., .2, .4, .6] if args.mode == 'interview' else 0.,
        word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    for audio in files:
        rel = audio.relative_to(wav_root).with_suffix('.json')
        segments, _ = model.transcribe(str(audio), **settings)
        segments = [dataclasses.asdict(s) for s in segments]
        text = ' '.join(s['text'].strip() for s in segments)
        result = dict(text=text, segments=segments, audio_output=str(audio.resolve()),
            engine='faster-whisper', engine_version=importlib.metadata.version('faster-whisper'),
            model='large-v3-turbo', device=args.device, settings=settings, review_status='automatic_draft')
        if args.mode == 'voice-check':
            reference = read(args.scripts/rel)['text']
            similarity = difflib.SequenceMatcher(None,norm(reference),norm(text),autojunk=False).ratio()
            result.update(reference_text=reference, asr_text=text, similarity=similarity,
                automatic_check_passed=similarity>=.95, screening_threshold=.95,
                manual_listening_performed=False)
        write(args.out/rel,result)
        print(rel)


if __name__ == '__main__':
    main()
