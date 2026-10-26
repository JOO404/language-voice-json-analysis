"""Aggregate reviewed interview analysis with the fixed study reference.

Auxiliary behavioural indicators, not participant Group A/B ground truth.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
from pipeline.common import read, write, csv_write
from ._core import analyze_bar_profile as algorithm


def fixed_scores(signals, reference):
    scores={}
    for key, ref in reference['dimensions'].items():
        value=signals.get(key)
        if value is None or not math.isfinite(value):
            scores[key]=None
        else:
            scores[key]=round(max(1.,min(5.,3.+(value-ref['mean'])/ref['sample_std'])),2) if ref['sample_std'] else 3.
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleaned',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True,help='Folder produced by analyze.py')
    parser.add_argument('--reference',type=Path,required=True,help='Fixed study reference JSON; never refit on test samples')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('Output must be new or empty')
    run=read(args.analysis/'run_manifest.json')
    if run['status']!='complete' or run['timing_mode']!='eligible' or run['intent_version']!='revised':
        parser.error('BAR requires complete interview analysis with eligible timing and final intent rules')
    if not {'emotion','intent','prosody'} <= set(run['stages']):
        parser.error('BAR requires emotion, intent and prosody stages')
    records=defaultdict(list)
    for path in sorted(args.cleaned.rglob('*.json')):
        rel=path.relative_to(args.cleaned)
        clip={'cleaned':read(path)}
        for stage in ['emotion','prosody','intent']:
            clip[stage]=read(args.analysis/'data'/stage/rel)
        person=clip['cleaned'].get('participant_id') or rel.parent.name
        for stage in ['emotion','intent']:
            if clip[stage].get('text','').strip()!=clip['cleaned'].get('text','').strip():
                raise ValueError(f'Mismatched cleaned text and {stage}: {rel}')
        records[person].append(clip)
    if not records:parser.error('No cleaned JSON files')
    reference=read(args.reference)
    rows=[]
    for person,clips in records.items():
        profile={'person':person,**algorithm.aggregate_person(person,clips)}
        scores=fixed_scores(profile['bar_signals'],reference)
        profile.update(bar_scores_1_5=scores,
            social_style=algorithm.classify_social_style(scores) if all(scores.get(k) is not None for k in ['dominance','directness','warmth','expressiveness']) else None,
            reference_method=reference['method'])
        write(args.out/'data/personality'/f'{person}.json',profile)
        row=dict(person=person,n_clips=profile['n_clips'],n_reliable_prosody=profile['n_prosody_reliable'],social_style=profile['social_style'],**scores,
            avg_utterance_length=round(profile['avg_utterance_length'],1),formal_ratio=round(profile['formal_ratio'],3))
        for label in ['statement','question','affirmation','negation','hesitation']:
            row['pct_'+label]=round(profile['intent_distribution'][label]*100,1)
        rows.append(row)
    csv_write(args.out/'personality_summary.csv',rows)


if __name__=='__main__':main()
