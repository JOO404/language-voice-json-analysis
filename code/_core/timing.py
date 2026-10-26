"""Conservative measurement eligibility, not an assertion of ASR error."""
import math
POLICY={'version':'uniform-timing-qc-1','minimum_duration_seconds':.064,
        'long_duration_seconds':1.,'minimum_long_word_vad_speech_fraction':.6,
        'note':'64 ms = two 32 ms pYIN hops. Operational sensitivity filter, not a validated linguistic threshold.'}

def inspect_word(w,clip):
    if w.get('type')!='word':return []
    if w.get('flag')=='overlap':return []  # separate, already excluded by original code
    a=w.get('start');b=w.get('end');reasons=[]
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (a,b)):
        return ['invalid_time']
    if b<=a:reasons.append('non_positive_duration')
    elif b-a<POLICY['minimum_duration_seconds']-1e-9:reasons.append('fewer_than_two_analysis_hops')
    if a<0 or (clip.get('duration') is not None and b>clip['duration']+.002):reasons.append('outside_audio')
    if b-a>=POLICY['long_duration_seconds'] and not clip.get('missing_audio'):
        share=sum(max(0,min(b,y)-max(a,x)) for x,y in clip['speech_intervals'])/(b-a)
        if share<POLICY['minimum_long_word_vad_speech_fraction']:reasons.append('long_interval_low_vad_speech')
    return reasons

def validate_override(entry,w,clip,source_hash,neighbors):
    if entry.get('source_sha256')!=source_hash:raise ValueError('Source changed; manual decision must be reviewed again.')
    action=entry.get('action')
    if action=='exclude':return None
    if action not in ('set','keep'):raise ValueError('Unknown timing decision')
    a,b=(entry.get('start'),entry.get('end')) if action=='set' else (w['start'],w['end'])
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (a,b)):raise ValueError('Times must be finite numbers')
    if clip.get('duration') is None or not 0<=a<b<=clip['duration']+.002:raise ValueError('Times must lie inside the audio')
    if b-a<POLICY['minimum_duration_seconds']-1e-9:raise ValueError('At least 0.064 seconds is needed for this acoustic analysis; otherwise keep excluded.')
    if action=='set':
        for other in neighbors:
            if other.get('type')=='word' and other.get('flag')!='overlap' and max(a,other['start'])<min(b,other['end'])-1e-6:
                raise ValueError('Corrected interval overlaps another retained word')
    return float(a),float(b)
