"""Voice and clip summaries, independent ASR checks, 64-ms sensitivity export.

Compare voices within the same group and question; A and B use different texts.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
from analyze import read, write, csv_write


def summarize(words):
    if not words:return dict(n_words=0,speech_duration=0.,speech_rate_wps=None,f0_mean=None,f0_std=None,rms_mean=None)
    duration=sum(w['duration'] for w in words)
    pitch=[w['f0_mean'] for w in words if w['f0_mean'] is not None]
    return dict(n_words=len(words),speech_duration=round(duration,3),
        speech_rate_wps=round(len(words)/duration,3) if duration>0 else None,
        f0_mean=round(float(np.mean(pitch)),2) if pitch else None,
        f0_std=round(float(np.std(pitch)),2) if len(pitch)>=2 else None,
        rms_mean=round(float(np.mean([w['rms_mean'] for w in words])),5))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--cleaned',type=Path,required=True)
    parser.add_argument('--asr',type=Path,help='Optional outputs of transcribe.py --mode voice-check')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('Output must be new or empty')
    source=args.analysis/'data/prosody'
    byvoice=defaultdict(list)
    clips=[];sensitivity=[];checks=[];timing=[]
    for path in sorted(source.rglob('*.json')):
        rel=path.relative_to(source)
        data=read(path);clean=read(args.cleaned/rel)
        fs=data.get('file_summary') or {}
        voice=clean.get('voice_id',data['person'])
        meta=dict(group=clean.get('group',''),voice_id=voice,voice_name=clean.get('voice_name',''),
            question_number=clean.get('question_number',rel.stem),file=rel.as_posix())
        row={**meta,'audio_duration_secs':clean.get('audio_duration_secs'),
            'f0_mean_hz':fs.get('f0_mean'),'f0_std_hz':fs.get('f0_std'),'rms_mean':fs.get('rms_mean'),
            'speech_rate_wps':fs.get('speech_rate_wps'),'qc_status':data.get('qc_status')}
        clips.append(row);byvoice[voice].append((data,meta))
        retained=[w for w in data.get('words',[]) if w['duration']>=.064-1e-9]
        sensitivity.append({**meta,**summarize(retained)})
        words=[w for w in clean['words'] if w.get('type')=='word']
        timing.append({**meta,'n_words':len(words),
            'short_words_under_64ms':sum(0<w['end']-w['start']<.064-1e-9 for w in words),
            'timing_review_required':sum(bool(w.get('timing_review_required')) for w in words),
            'manual_review':clean.get('review_status')=='human_reviewed'})
        if args.asr:
            check=read(args.asr/rel)
            checks.append({**meta,'normalized_similarity':round(check['similarity'],4),
                'automatic_check_passed':check['similarity']>=.95,'script':clean['text'],
                'asr_text':check.get('asr_text',check['text']),'manual_listening':False})
    if not clips:parser.error('No prosody JSON files found')
    voices=[]
    for voice,items in byvoice.items():
        ds=[d for d,_ in items];healthy=[d for d in ds if d.get('prosody_reliable')]
        meta=items[0][1]
        row=dict(group=meta['group'],voice_id=voice,voice_name=meta['voice_name'],n_clips=len(ds),
            healthy_clips=len(healthy),suspect_or_failed_clips=len(ds)-len(healthy))
        for key,outkey,nd in [('f0_mean','f0_mean_hz',2),('f0_std','within_clip_f0_std_hz',2),('rms_mean','rms_mean',5),('speech_rate_wps','speech_rate_wps',3)]:
            for subset,label in [(ds,'all'),(healthy,'healthy')]:
                values=[d['file_summary'][key] for d in subset if d.get('file_summary') and d['file_summary'].get(key) is not None]
                row[outkey+'_'+label]=round(statistics.mean(values),nd) if values else None
        voices.append(row)
    args.out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('clip_comparison',clips),('voice_comparison',voices),('timing_sensitivity',sensitivity),('timing_qc',timing),('script_audio_check',checks)]:
        csv_write(args.out/(name+'.csv'),rows)
    write(args.out/'summary.json',dict(n_clips=len(clips),n_voices=len(voices),
        comparison_scope='Within group and question; between-group wording differs.',
        script_check_threshold=.95 if args.asr else None,
        script_check_note='Screening heuristic, not manual verification or a validated accuracy threshold.'))


if __name__=='__main__':main()
