"""Final language and voice analysis entry point; calculations are in _core/.

Runs cleaned-JSON -> prosody/QC, intent and optionally text emotion.
Does not transcribe, align, annotate, assign research groups or fit BAR norms.
"""
import argparse
import csv
import os
from pathlib import Path
import tempfile

from pipeline.common import read, write, csv_write


def prepare_inputs(project, timing_mode, cleaned=None):
    source = cleaned or project / 'data/cleaned'
    files = sorted(source.rglob('*.json'))
    if not files:
        raise ValueError('No JSON files under project/data/cleaned.')
    entries = []
    for path in files:
        rel = path.relative_to(source)
        if len(rel.parts) != 2:
            raise ValueError(f'Expected participant_id/clip.json: {rel}')
        data = read(path)
        if not isinstance(data, dict) or not isinstance(data.get('words'), list):
            raise ValueError(f'Expected an object with words[]: {rel}')
        if not isinstance(data.get('text', ''), str):
            raise ValueError(f'text must be a string: {rel}')
        if 'text_reliable' in data and not isinstance(data['text_reliable'], bool):
            raise ValueError(f'text_reliable must be a Boolean: {rel}')
        words = data['words']
        excluded = 0
        if timing_mode == 'eligible':
            # Consume already reviewed V3 flags; never infer/recreate timing QC here.
            for word in words:
                if word.get('type') == 'word' and word.get('flag') != 'overlap':
                    if not isinstance(word.get('acoustic_eligible'), bool):
                        raise ValueError(f'Missing Boolean acoustic_eligible: {rel}')
            selected = [w for w in words if w.get('type') != 'word'
                        or w.get('flag') == 'overlap' or w['acoustic_eligible']]
            excluded = len(words) - len(selected)
            data = {**data, 'words': selected}
        entries.append((rel, data, excluded))
    return entries


def run_prosody(project, out, entries, timing_mode):
    import librosa
    from ._core import analyze_prosody as algorithm
    from ._core import qc_prosody as qc
    dest = out / 'data/prosody'
    dest.mkdir(parents=True, exist_ok=True)
    qc.PROSODY_DIR = dest
    qc.QC_CSV = out / 'prosody_qc_report.csv'
    qc.QC_SUMMARY = out / 'prosody_qc_summary.txt'
    # Copies are temporary; all source JSON/WAV files are read-only inputs.
    with tempfile.TemporaryDirectory(prefix='_working_', dir=out) as temporary:
        temp = Path(temporary)
        for rel, data, excluded in entries:
            write(temp / rel, data)

        def analyze(rel, data, excluded, fmin, fmax):
            result, error = algorithm.analyze_file(temp / rel,
                project / 'data/wav' / rel.with_suffix('.wav'), fmin, fmax, librosa)
            if result is None:
                result = {'skip_reason': error, 'file_summary': None, 'words': []}
            # Retain full identifiers (A1/B1 must never be merged as "1").
            result.update(person=data.get('participant_id') or data.get('voice_id') or rel.parent.name, code=rel.stem)
            for field in ['voice_id', 'voice_name', 'group', 'question_number', 'source_question_code', 'timing_quality']:
                if field in data:
                    result[field] = data[field]
            if timing_mode == 'eligible':
                result['timing_excluded_words'] = excluded
                result['timing_policy'] = 'precomputed_acoustic_eligible_flags'
            write(dest / rel, result)

        for rel, data, excluded in entries:
            print('Prosody:', rel)
            analyze(rel, data, excluded, 65, 350)
        qc.main()
        with qc.QC_CSV.open(encoding='utf-8-sig', newline='') as stream:
            retry = {r['file'].replace('\\', '/') for r in csv.DictReader(stream)
                     if r['status'] != 'healthy'}
        for rel, data, excluded in entries:
            if rel.as_posix() in retry:
                analyze(rel, data, excluded, 80, 300)
        if retry:
            qc.main()

    with qc.QC_CSV.open(encoding='utf-8-sig', newline='') as stream:
        qc_rows = list(csv.DictReader(stream))
    for row in qc_rows:
        path = dest / Path(row['file'].replace('\\', '/'))
        data = read(path)
        data.update(prosody_reliable=row['status'] == 'healthy',
                    prosody_usable=row['status'] != 'failed', qc_status=row['status'])
        if row['issues']:
            data['qc_issues'] = row['issues']
        write(path, data)

    # Export all final files, including files that did not need a second pass.
    file_rows, word_rows = [], []
    for rel, _, _ in entries:
        data = read(dest / rel)
        fs = data.get('file_summary')
        if not fs:
            continue
        base = dict(file=rel.as_posix(), person=data['person'], code=data['code'])
        file_rows.append({**base, 'n_words': fs['num_words_analyzed'],
            'excluded_overlap': fs['num_words_excluded_overlap'],
            **{k: fs[k] for k in ['speech_duration', 'speech_rate_wps', 'f0_mean', 'f0_std', 'rms_mean']}})
        for word in data['words']:
            word_rows.append({**base, 'word_idx': word['index'], 'word_text': word['text'],
                **{k: word[k] for k in ['start', 'end', 'duration', 'f0_mean', 'f0_std',
                    'f0_range', 'f0_slope', 'rms_mean', 'rms_max', 'voiced_ratio', 'n_voiced_frames']}})
    csv_write(out / 'prosody_file_summary.csv', file_rows)
    csv_write(out / 'prosody_word_summary.csv', word_rows)


def run_language(out, entries, stages, intent_version, emotion_model):
    from ._core import intent_rules, detect_intent
    intent = intent_rules if intent_version == 'revised' else detect_intent
    pipe = None
    if 'emotion' in stages:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from ._core import analyze_emotion
        torch.set_num_threads(4)
        analyze_emotion.MODEL_ID = str(emotion_model)
        pipe = analyze_emotion.load_pipeline()
    intent_rows, emotion_rows = [], []
    for rel, data, _ in entries:
        text = data.get('text', '').strip()
        reliable = data.get('text_reliable', True)
        base = dict(person=data.get('participant_id') or data.get('voice_id') or rel.parent.name, code=rel.stem, text=text, text_reliable=reliable)
        skip = 'text_reliable=false' if not reliable else 'empty text' if not text else None
        if 'intent' in stages:
            pd = out / 'data/prosody' / rel
            prosody = read(pd) if pd.exists() else None
            if skip:
                result = {**base, 'skip_reason': skip, 'primary_intent': None}
            else:
                primary, scores, signals, confidence = intent.detect_intent(text, prosody)
                mode, mode_signals = intent.detect_speech_mode(text)
                result = {**base, 'primary_intent': primary, 'confidence': confidence,
                    'speech_mode': mode, 'scores': {k: round(v, 2) for k, v in scores.items()},
                    'signals': signals, 'speech_mode_signals': mode_signals,
                    'used_prosody': bool(prosody and prosody.get('prosody_reliable', False))}
                if intent_version == 'revised':
                    result.update(rule_version=intent.VERSION, prosody_decision_role='supporting_evidence_only')
                intent_rows.append(dict(file=rel.as_posix(), person=base['person'], code=base['code'],
                    text=text, primary_intent=primary, speech_mode=mode, confidence=confidence,
                    n_signals=len(signals), used_prosody=result['used_prosody']))
            write(out / 'data/intent' / rel, result)
        if pipe is not None:
            if skip:
                result = {**base, 'skip_reason': skip, 'top_emotions': [], 'all_scores': {}}
            else:
                scores = sorted(pipe(text)[0], key=lambda r: r['score'], reverse=True)
                result = {**base, 'top_emotions': [{'label': s['label'], 'score': round(s['score'], 4)} for s in scores[:5]],
                    'all_scores': {s['label']: round(s['score'], 4) for s in scores}}
                row = dict(file=rel.as_posix(), person=base['person'], code=base['code'], text=text)
                for i, score in enumerate(scores[:5], 1):
                    row[f'top{i}_label'], row[f'top{i}_score'] = score['label'], round(score['score'], 4)
                emotion_rows.append(row)
            write(out / 'data/emotion' / rel, result)
    csv_write(out / 'intent_summary.csv', intent_rows)
    csv_write(out / 'emotion_summary.csv', emotion_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cleaned', type=Path, help='Optional separate folder of reviewed/timing-qualified JSON')
    parser.add_argument('--stages', default='prosody,intent')
    parser.add_argument('--timing-mode', choices=['positive', 'eligible'], default='positive')
    parser.set_defaults(intent_version='revised')
    parser.add_argument('--emotion-model', type=Path, help='Local KOTE model folder; no downloads are performed.')
    args = parser.parse_args(argv)
    stages = set(args.stages.split(','))
    if not stages or not stages <= {'prosody', 'intent', 'emotion'}:
        parser.error('stages must be comma-separated prosody,intent,emotion')
    project, out = args.project.resolve(), args.out.resolve()
    if out == project or project.is_relative_to(out) or out.is_relative_to(project / 'data'):
        parser.error('Use a separate output directory outside input data/.')
    if args.cleaned and (out == args.cleaned.resolve() or out.is_relative_to(args.cleaned.resolve()) or args.cleaned.resolve().is_relative_to(out)):
        parser.error('Output must be separate from --cleaned input.')
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be new or empty; existing results are never overwritten.')
    if 'emotion' in stages and (not args.emotion_model or not args.emotion_model.is_dir()):
        parser.error('emotion requires --emotion-model with a local KOTE model directory')
    entries = prepare_inputs(project, args.timing_mode, args.cleaned)
    if 'prosody' in stages:
        missing = [str(rel) for rel, _, _ in entries if not (project / 'data/wav' / rel.with_suffix('.wav')).is_file()]
        if missing:
            parser.error(f'Missing WAV inputs: {missing[:5]}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(adapter_version='handoff-1', status='running', stages=sorted(stages),
        n_inputs=len(entries), timing_mode=args.timing_mode, intent_version=args.intent_version,
        source_project=str(project), emotion_model=str(args.emotion_model) if args.emotion_model else None,
        note='Input files unchanged. Final study calculations with configurable input/output paths.')
    write(out / 'run_manifest.json', manifest)
    try:
        if 'prosody' in stages:
            run_prosody(project, out, entries, args.timing_mode)
        if stages & {'intent', 'emotion'}:
            run_language(out, entries, stages, args.intent_version, args.emotion_model)
        manifest['status'] = 'complete'
        if 'prosody' in stages:
            manifest['n_prosody_skipped'] = sum(bool(read(out / 'data/prosody' / rel).get('skip_reason')) for rel, _, _ in entries)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        write(out / 'run_manifest.json', manifest)
        raise
    write(out / 'run_manifest.json', manifest)
    print('Saved:', out)


if __name__ == '__main__':
    main()
