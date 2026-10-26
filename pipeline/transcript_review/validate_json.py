"""Read-only structural/timing check for raw/formatted/cleaned word JSON.

This does not establish transcription accuracy or replace listening review.
Exit 1 means structural errors; warnings alone do not fail validation.
"""
import argparse
import json
import math
from pathlib import Path


def check(data):
    errors, warnings = [], []
    if not isinstance(data, dict):
        return ['Root must be an object'], []
    if not isinstance(data.get('text'), str):
        errors.append('text must be a string')
    if not isinstance(data.get('words'), list):
        return errors + ['words must be an array'], warnings
    if 'text_reliable' in data and not isinstance(data['text_reliable'], bool):
        errors.append('text_reliable must be Boolean')
    duration = data.get('audio_duration_secs')
    number = lambda value: type(value) in (int, float) and math.isfinite(value)
    if duration is not None and (not number(duration) or duration <= 0):
        errors.append('audio_duration_secs must be a finite positive number')
        duration = None
    previous = None
    for i, word in enumerate(data['words']):
        prefix = f'words[{i}]'
        if not isinstance(word, dict):
            errors.append(prefix + ' must be an object')
            continue
        if word.get('type') not in ['word', 'spacing']:
            errors.append(prefix + ' type must be word or spacing')
        if not isinstance(word.get('text'), str):
            errors.append(prefix + ' text must be a string')
        if word.get('type') != 'word':
            continue
        start, end = word.get('start'), word.get('end')
        if not number(start) or not number(end):
            errors.append(prefix + ' timestamps must be finite numbers')
            continue
        excluded = word.get('flag') == 'overlap' or word.get('acoustic_eligible') is False
        destination = warnings if excluded else errors
        if start < 0 or end < start or (duration is not None and end > duration + .005):
            destination.append(prefix + ' invalid/out-of-bounds interval' + (' (already excluded)' if excluded else ''))
        if end == start:
            warnings.append(prefix + ' has zero duration; original prosody skips it')
        elif end - start < .064:
            warnings.append(prefix + ' shorter than two 32-ms analysis hops')
        if previous is not None and start < previous - .000001:
            warnings.append(prefix + ' overlaps the preceding word interval')
        previous = end
        if 'acoustic_eligible' in word and not isinstance(word['acoustic_eligible'], bool):
            errors.append(prefix + ' acoustic_eligible must be Boolean')
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    files = [args.source] if args.source.is_file() else sorted(args.source.rglob('*.json'))
    report = {'n_files': len(files), 'errors': 0, 'warnings': 0, 'files': []}
    for path in files:
        try:
            errors, warnings = check(json.loads(path.read_text(encoding='utf-8-sig')))
        except Exception as exc:
            errors, warnings = [str(exc)], []
        report['files'].append({'file': str(path), 'errors': errors, 'warnings': warnings})
        report['errors'] += len(errors)
        report['warnings'] += len(warnings)
    if not files:
        report['errors'] += 1
        report['message'] = 'No input JSON files found'
    if args.report:
        if args.report.exists():
            parser.error('Report already exists; choose a new path.')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
