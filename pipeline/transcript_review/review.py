"""Small Windows adapter around the unchanged original annotation code."""
from pathlib import Path
import argparse
import builtins
import csv
import datetime
import json
import msvcrt
import os
import shutil
import sys
import winsound

CODE_DIR=Path(__file__).resolve().parent
BASE=CODE_DIR
from ._core import annotate_words as legacy
legacy.DATA_DIR=BASE/'data/formatted'
legacy.OUTPUT_DIR=BASE/'data/cleaned'
legacy.PROGRESS_CSV=BASE/'cleanup_progress.csv'
legacy.MAP_CSV=BASE/'video_map.csv'

_active_rel=None
_audio_map=None
_original_input=builtins.input

def audio_map():
    global _audio_map
    if _audio_map is None:
        _audio_map={str(p.relative_to(legacy.DATA_DIR)):str(BASE/'data/wav'/p.relative_to(legacy.DATA_DIR).with_suffix('.wav')) for p in legacy.DATA_DIR.rglob('*.json')}
    return _audio_map

def stop_audio():
    winsound.PlaySound(None,0)

def play_audio(rel):
    path=audio_map().get(rel)
    if not path or not Path(path).is_file():
        print('  [재생 오류] 연결된 음성 파일을 찾을 수 없습니다.')
        return
    try:
        stop_audio()
        winsound.PlaySound(path,winsound.SND_FILENAME|winsound.SND_ASYNC|winsound.SND_NODEFAULT)
    except RuntimeError as exc:
        print(f'  [재생 오류] {exc}. v 입력으로 영상을 열어 확인할 수 있습니다.')

def review_input(prompt=''):
    while True:
        result=_original_input(prompt)
        if prompt=='  > ' and result.strip().lower()=='v':
            path=legacy.load_video_map().get(_active_rel)
            if path and Path(path).is_file():
                stop_audio()
                os.startfile(path)
                print('  영상을 기본 재생 앱으로 열었습니다. 명령은 이 터미널에서 입력하세요.')
            else:
                print('  연결된 영상이 없습니다.')
            continue
        return result

def print_help():
    print('\n  Enter       저장하고 다음 답변 (음성 자동 재생)')
    print('  4b / 4o     4번 전사 오류 / 음성 겹침 표시 후 다음')
    print('  4 / 4-8     4번 / 4~8번 삭제 후 다음 (인터뷰어 발화 등)')
    print('  2,4-8o,10b  여러 항목을 한 번에 지정')
    print('  p           처음부터 다시 듣기 (원래 속도)')
    print('  v           영상 열기 (필요할 때만)')
    print('  B           이 답변 전체의 전사문을 불량으로 표시 후 다음')
    print('  g / s       파일로 이동·재검토 / 현재 답변 건너뛰기')
    print('  q           현재 답변은 저장하지 않고 종료 (앞선 저장 결과 유지)')
    print('  ?           도움말\n')
    print('  모든 명령은 입력 후 Enter. 번호는 화면의 [번호]를 사용합니다.')

def backup_and_reset(rel):
    cleaned=legacy.OUTPUT_DIR/rel
    if cleaned.exists():
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dst=BASE/'review_backups'/stamp/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(cleaned,dst)
    legacy.reset_file(rel)

def save_review_metadata(rel):
    dst=legacy.OUTPUT_DIR/rel
    data=json.loads(dst.read_text(encoding='utf-8'))
    data['review_status']='human_reviewed'
    data['reviewed_at']=datetime.datetime.now().isoformat(timespec='seconds')
    data['review_tool']='original annotate_words.py + Windows audio adapter'
    data['remaining_timing_review_indices']=[i for i,w in enumerate(data['words'])
        if w.get('type')=='word' and (w.get('timing_review_required') or w['end']<=w['start'])]
    temporary=dst.with_suffix('.json.tmp')
    temporary.write_text(legacy.smart_format(data),encoding='utf-8')
    temporary.replace(dst)

def choose_target(rel_list,query):
    matches=[r for r in rel_list if query in r]
    if not matches:
        print('  일치하는 파일이 없습니다.');return None
    if len(matches)==1:return matches[0]
    for i,m in enumerate(matches[:20]):print(f'  [{i}] {m}')
    choice=_original_input('  이동할 번호 (Enter=취소): ').strip()
    if choice.isdigit() and int(choice)<min(20,len(matches)):
        return matches[int(choice)]
    return None

def main():
    global _active_rel, BASE
    parser=argparse.ArgumentParser(description='Keyboard transcript review with automatic WAV playback (Windows).')
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args()
    BASE=args.project.resolve()
    legacy.DATA_DIR=BASE/'data/formatted'
    legacy.OUTPUT_DIR=BASE/'data/cleaned'
    legacy.PROGRESS_CSV=BASE/'cleanup_progress.csv'
    legacy.MAP_CSV=BASE/'video_map.csv'
    if not legacy.DATA_DIR.is_dir():raise ValueError('Missing data/formatted')
    lock=(BASE/'review.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        print('이미 검토 창이 열려 있습니다. 그 창에서 이어서 진행하세요.')
        lock.close();return
    legacy.open_video=play_audio
    legacy.print_help=print_help
    builtins.input=review_input
    try:
        all_files=sorted(legacy.DATA_DIR.rglob('*.json'))
        rel_list=[str(f.relative_to(legacy.DATA_DIR)) for f in all_files]
        # CSV and saved JSON must agree; a missing/incomplete result is reviewed again.
        processed=set()
        for rel in legacy.load_progress():
            path=legacy.OUTPUT_DIR/rel
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('review_status')=='human_reviewed':
                processed.add(rel)
        print('인터뷰 전사 검토 / Transcript review')
        print(f'전체 {len(all_files)}개 | 완료 {len(processed)}개 | 남음 {len(all_files)-len(processed)}개')
        print_help()
        idx=0
        while idx<len(all_files):
            rel=rel_list[idx]
            if rel in processed:idx+=1;continue
            _active_rel=rel
            data=json.loads(all_files[idx].read_text(encoding='utf-8'))
            print('\n'+'='*68)
            print(f"[{idx+1}/{len(all_files)} · 완료 {len(processed)}개] {data.get('participant_name',all_files[idx].parent.name)} ({data.get('participant_id','')}) / {data.get('clip_code',all_files[idx].stem)} / {data.get('question_label','')}")
            print(f'  파일: {rel}')
            notes=data.get('review_notes',[])
            for note in notes:print(f'  참고: {note}')
            play_audio(rel)
            result,payload=legacy.process_file(all_files[idx],rel)
            stop_audio()
            if result=='quit':break
            if result=='done':
                save_review_metadata(rel)
                processed.add(rel)
                idx+=1
            elif result=='skip':idx+=1
            elif result=='goto':
                target=choose_target(rel_list,payload)
                if target:
                    backup_and_reset(target)
                    processed.discard(target)
                    idx=rel_list.index(target)
                    print('  이전 검토본을 백업하고 해당 답변을 원본 전사부터 다시 엽니다.')
        summary={'reviewed':len(processed),'total':len(all_files),'complete':len(processed)==len(all_files),
                 'updated_at':datetime.datetime.now().isoformat(timespec='seconds')}
        (BASE/'review_status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"\n검토 저장: {len(processed)}/{len(all_files)}개")
        print(f'결과 폴더: {legacy.OUTPUT_DIR}')
        if summary['complete']:print('검토가 모두 끝났습니다. 이 새 검토 결과로 후속 분석을 진행할 수 있습니다.')
        else:print('같은 실행 파일을 열면 아직 저장하지 않은 답변부터 이어집니다.')
    except (KeyboardInterrupt,EOFError):
        print('\n종료했습니다. 이미 저장한 검토 결과는 유지됩니다.')
    finally:
        stop_audio()
        builtins.input=_original_input
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        lock.close()

if __name__=='__main__':main()
