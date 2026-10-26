"""Korean word alignment calculations."""
import math
import soundfile as sf
import torch
from whisperx.alignment import align

class ModelInputNormalization(torch.nn.Module):
    """Honor do_normalize=True in the aligner's preprocessor_config.json.

    Only the aligner's in-memory input is normalized. The original WAV and
    downstream acoustic measurements use the original recorded signal.
    """
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        x=(x-x.mean(dim=-1,keepdim=True))/torch.sqrt(x.var(dim=-1,keepdim=True,unbiased=False)+1e-7)
        return self.model(x)

def valid(w,duration):
    return all(isinstance(w.get(k),(int,float)) and math.isfinite(w[k]) for k in ('start','end')) and 0<=w['start']<w['end']<=duration+0.002

def align_one(segments,audio_path,model,metadata):
    y,sr=sf.read(str(audio_path),dtype='float32')
    assert sr==16000 and y.ndim==1
    duration=len(y)/sr
    # A token-sized ASR segment (e.g. a separate 40 ms '네.') can be shorter
    # than the model can align. Merge only these with an adjacent predecessor.
    grouped=[]
    for s in segments:
        s=dict(s)
        s['words']=list(s['words'])
        if grouped and s['end']-s['start']<0.25 and s['start']-grouped[-1]['end']<0.5 and s['end']-grouped[-1]['start']<30:
            grouped[-1]['end']=s['end']
            grouped[-1]['text']+=' '+s['text'].strip()
            grouped[-1]['words']+=s['words']
        else:grouped.append(s)
    segments=grouped
    results=[]
    fallbacks=[]
    for si,s in enumerate(segments):
        text=' '.join(s['text'].split())
        if not text:
            continue
        start=max(0.,min(float(s['start']),duration))
        end=min(duration,max(float(s['end']),start))
        if si==0:
            start=0.0
        if si==len(segments)-1:
            end=duration
        if end-start<0.026:
            start=max(0,end-0.026)
        aligned=align([dict(start=start,end=end,text=text)],model,metadata,y,'cpu',return_char_alignments=True)
        words=aligned['word_segments']
        if [w['word'] for w in words]!=text.split():
            # Never omit text when forced alignment fails. Keep ASR timings,
            # explicitly flagged for review and excluded by downstream timing QC.
            fallbacks.append(si)
            words=[dict(word=w['word'].strip(),start=w['start'],end=w['end'],
                        asr_probability=w['probability'],timing_source='asr_fallback',
                        timing_review_required=True) for w in s['words']]
        else:
            for w in words:
                w['timing_source']='whisperx_forced_alignment'
                if not valid(w,duration) or 'score' not in w or (w['end']-w['start']<=0.041 and w.get('score',0)<0.10):
                    w['timing_review_required']=True
        results.extend(words)
    stats={
        'n_words':len(results),
        'duration_sum':round(sum(w['end']-w['start'] for w in results if valid(w,duration)),6),
        'invalid_timing_indices':[i for i,w in enumerate(results) if not valid(w,duration)],
        'review_timing_indices':[i for i,w in enumerate(results) if w.get('timing_review_required')],
        'low_score_indices':[i for i,w in enumerate(results) if w.get('score',1)<0.30],
        'overlap_indices':[i for i in range(1,len(results)) if results[i].get('start',0)<results[i-1].get('end',0)-.002],
        'fallback_segments':fallbacks,
    }
    return dict(words=results,stats=stats,audio_duration_secs=duration)
