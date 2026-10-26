"""Local Korean word alignment for interview ASR or supplied voice scripts.

Outputs formatted JSON for human review; acoustic WAV is never normalized.
"""
import argparse
import math
import os
from pathlib import Path
from pipeline.common import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='ASR JSON folder or supplied script JSON folder')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='Local Korean alignment model directory')
    parser.add_argument('--nltk-data', type=Path)
    parser.add_argument('--mode', choices=['interview','script'], default='interview')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output must be new or empty')
    if not args.model.is_dir():
        parser.error('Local alignment model not found')
    files = sorted(args.input.rglob('*.json'))
    if not files:
        parser.error('No input JSON found')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    if args.nltk_data:
        os.environ['NLTK_DATA']=str(args.nltk_data.resolve())
    import torch
    import soundfile as sf
    from whisperx.alignment import load_align_model, align
    from ._core.alignment import ModelInputNormalization, align_one, valid
    torch.set_num_threads(4)
    model, meta = load_align_model('ko','cpu',model_name=str(args.model.resolve()),model_cache_only=True)
    model = ModelInputNormalization(model).eval()
    for path in files:
        rel = path.relative_to(args.input)
        source = read(path)
        audio = args.project/'data/wav'/rel.with_suffix('.wav')
        if args.mode == 'interview':
            aligned = align_one(source['segments'],audio,model,meta)
            words, duration = aligned['words'], aligned['audio_duration_secs']
        else:
            y, sr = sf.read(str(audio),dtype='float32')
            if sr != 16000 or y.ndim != 1:
                raise ValueError('Expected 16-kHz mono WAV')
            duration = len(y)/sr
            with torch.inference_mode():
                aligned = align([dict(start=0.,end=duration,text=source['text'])],model,meta,y,'cpu',
                    return_char_alignments=True,interpolate_method='ignore')
            words = aligned['word_segments']
            if [w['word'] for w in words] != source['text'].split():
                raise ValueError(f'Alignment token mismatch: {rel}; human review required')
        output_words=[]
        for i, word in enumerate(words):
            issues=[]
            good=valid(word,duration)
            if not good:
                issues.append('invalid_time')
            else:
                if word['end']-word['start'] < .064-1e-9:issues.append('fewer_than_two_analysis_hops')
                if word.get('score',1)<.3:issues.append('low_alignment_score')
                if output_words and word['start']<output_words[-1]['end']-.002:issues.append('overlapping_word_times')
            # Explicit nonmeasurable placeholder; never invent a positive interval.
            start,end=(word['start'],word['end']) if good else (0.,0.)
            output_words.append(dict(text=word['word'].strip(),start=start,end=end,type='word',flag='',
                alignment_score=word.get('score'),timing_source=word.get('timing_source','whisperx_forced_alignment'),
                timing_review_required=bool(issues) or word.get('timing_review_required',False),timing_issues=issues))
        result=dict(text=source['text'],words=output_words,audio_duration_secs=duration,
            text_reliable=True,review_status='automatic_draft',manual_listening_performed=False,
            alignment_engine='WhisperX',alignment_model_input_normalized=True,acoustic_audio_normalized=False)
        for field in ['participant_id','participant_name','voice_id','voice_name','group','question_number','clip_code','question_label']:
            if field in source:result[field]=source[field]
        write(args.out/rel,result)
        print(rel)


if __name__ == '__main__':
    main()
