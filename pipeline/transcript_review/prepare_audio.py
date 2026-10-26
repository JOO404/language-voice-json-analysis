"""Convert supplied clips or explicit reviewed video intervals to PCM16 WAV.

The job list assigns IDs and boundaries; this does not detect interview questions.
"""
import argparse
import math
from pathlib import Path
import subprocess
from pipeline.common import read, write


def component(value):
    value=str(value)
    if not value or any(c in value for c in '/\\:*?"<>|') or value in ['.','..']:
        raise ValueError('IDs must be single filename components')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True,help='JSON array: input, participant_id/voice_id, clip_code, optional start/end/text')
    parser.add_argument('--project',type=Path,required=True,help='New project directory')
    parser.add_argument('--ffmpeg',default='ffmpeg')
    args=parser.parse_args()
    if args.project.exists() and any(args.project.iterdir()):parser.error('Project must be new or empty')
    jobs=read(args.jobs)
    if not isinstance(jobs,list) or not jobs:parser.error('Expected a nonempty job array')
    planned=[];seen=set()
    for job in jobs:
        person=component(job.get('participant_id') or job.get('voice_id') or '')
        clip=component(job['clip_code'])
        rel=Path(person)/(clip+'.wav')
        if rel in seen:raise ValueError('Duplicate output: '+str(rel))
        seen.add(rel)
        source=Path(job['input'])
        if not source.is_absolute():source=args.jobs.resolve().parent/source
        if not source.is_file():raise ValueError('Input not found: '+str(source))
        start,end=job.get('start'),job.get('end')
        if (start is None)!=(end is None):raise ValueError('Both start and end must be supplied together')
        if start is not None and not (all(type(v) in (int,float) and math.isfinite(v) for v in [start,end]) and 0<=start<end):
            raise ValueError('Invalid start/end in seconds')
        planned.append((job,rel,source,start,end))
    for job,rel,source,start,end in planned:
        destination=args.project/'data/wav'/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        command=[args.ffmpeg,'-n','-i',str(source)]
        if start is not None:command+=['-ss',str(start),'-t',str(end-start)]
        command+=['-vn','-ar','16000','-ac','1','-c:a','pcm_s16le','-loglevel','error',str(destination)]
        subprocess.run(command,check=True)
        if 'text' in job:
            write(args.project/'data/scripts'/rel.with_suffix('.json'),{k:v for k,v in job.items() if k not in ['input','start','end']})
    write(args.project/'audio_jobs.json',jobs)


if __name__=='__main__':main()
