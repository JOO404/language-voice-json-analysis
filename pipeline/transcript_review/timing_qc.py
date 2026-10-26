"""Apply the final interview timing eligibility policy to reviewed JSON copies."""
import argparse
import hashlib
from pathlib import Path
import wave
from pipeline.common import read, write
from ._core.timing import POLICY, inspect_word, validate_override


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='New folder for timing-qualified cleaned JSON')
    parser.add_argument('--decisions',type=Path,help='Optional hash-bound manual timing decisions keyed by participant/clip.json#index')
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('Output must be new or empty')
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    source=args.project/'data/cleaned'
    files=sorted(source.rglob('*.json'))
    if not files:parser.error('No cleaned JSON inputs')
    decisions=read(args.decisions) if args.decisions else {}
    for path in files:
        rel=path.relative_to(source)
        with wave.open(str(args.project/'data/wav'/rel.with_suffix('.wav')),'rb') as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):
                raise ValueError('Expected PCM16 mono 16-kHz WAV')
            frames=f.getnframes()
            audio=np.frombuffer(f.readframes(frames),dtype='<i2').astype('float32')/32768
        spans=get_speech_timestamps(audio,VadOptions(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=100,speech_pad_ms=0))
        clip={'duration':frames/16000,'speech_intervals':[[s['start']/16000,s['end']/16000] for s in spans]}
        data=read(path)
        source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        for i,word in enumerate(data['words']):
            if word.get('type')!='word':continue
            reasons=inspect_word(word,clip)
            decision=decisions.get(f'{rel.as_posix()}#{i}')
            if decision:
                interval=validate_override(decision,word,clip,source_hash,[w for j,w in enumerate(data['words']) if j!=i])
                if interval is None:reasons=['manual_acoustic_exclusion']
                else:
                    word['start'],word['end']=interval
                    reasons=[]
            if word.get('flag')=='overlap':reasons=['overlap']
            word['acoustic_eligible']=not reasons
            word['acoustic_exclusion_reasons']=reasons
        data['acoustic_policy_version']=POLICY['version']
        write(args.out/rel,data)
        print(rel)


if __name__=='__main__':main()
