#!/usr/bin/env python3
"""
JSON 정리 + 플래그 통합 도구 (v3 - 매핑 기반)
=============================================
처리 종류:
  - 삭제      : 인터뷰어/환각/틀린 어노테이션
  - overlap   : 음성 겹침 (prosody 분석 시 제외)
  - bad_text  : 외국어 오전사 등 (텍스트 분석 시 제외)
  - 파일 전체 : text_reliable=false

영상 연결: build_video_map.py로 만든 video_map.csv를 읽어서 p 키로 재생.
           (런타임 매칭 안 함 - 미리 만든 매핑만 사용)

원본(data/formatted)은 절대 수정 안 함. 결과는 data/cleaned 에만.

사용:
  1) python build_video_map.py     (먼저 매핑 생성, 한 번만)
  2) python annotate_words.py
"""

import json
import csv
import re
import datetime
import subprocess
import platform
from pathlib import Path

# ====== 설정 ======
DATA_DIR = Path("data/formatted")
OUTPUT_DIR = Path("data/cleaned")
PROGRESS_CSV = Path("cleanup_progress.csv")
MAP_CSV = Path("video_map.csv")     # build_video_map.py가 만든 매핑

VIDEO_PLAYER = "vlc"        # "vlc"(배속 지원) 또는 "default"(OS 기본 앱)
VIDEO_RATE = 1.5            # 재생 배속 (VLC 전용)
VLC_PATHS = {
    "Darwin": "/Applications/VLC.app/Contents/MacOS/VLC",
    "Windows": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "Linux": "vlc",
}
# ==================

COMPACT_ARRAY_KEYS = {"words"}
PROGRESS_FIELDS = ['file', 'deleted', 'overlap', 'bad_text', 'text_reliable', 'timestamp']

KOREAN_RE = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
FOREIGN_LETTER_RE = re.compile(r'[a-zA-Zà-ÿ\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]')


def smart_format(data, indent=2):
    def fmt(val, depth=0, parent_key=None):
        pad = " " * (indent * depth)
        inner_pad = " " * (indent * (depth + 1))
        if isinstance(val, list) and parent_key in COMPACT_ARRAY_KEYS and val:
            parts = [f"{inner_pad}{json.dumps(item, ensure_ascii=False)}" for item in val]
            return "[\n" + ",\n".join(parts) + f"\n{pad}]"
        if isinstance(val, dict):
            if not val: return "{}"
            parts = []
            for k, v in val.items():
                k_json = json.dumps(k, ensure_ascii=False)
                v_str = fmt(v, depth + 1, parent_key=k)
                parts.append(f"{inner_pad}{k_json}: {v_str}")
            return "{\n" + ",\n".join(parts) + f"\n{pad}}}"
        if isinstance(val, list):
            if not val: return "[]"
            parts = [f"{inner_pad}{fmt(item, depth + 1)}" for item in val]
            return "[\n" + ",\n".join(parts) + f"\n{pad}]"
        return json.dumps(val, ensure_ascii=False)
    return fmt(data, 0)


def is_likely_foreign(text):
    s = text.strip()
    if not s:
        return False
    return bool(FOREIGN_LETTER_RE.search(s)) and not bool(KOREAN_RE.search(s))


# ---- 영상 매핑 ----
_video_map = None

def load_video_map():
    """video_map.csv -> {json_rel: video_path}"""
    global _video_map
    if _video_map is not None:
        return _video_map
    _video_map = {}
    if MAP_CSV.exists():
        with open(MAP_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('video'):
                    _video_map[row['json']] = row['video']
    return _video_map


def video_summary():
    vmap = load_video_map()
    if not MAP_CSV.exists():
        print("  (video_map.csv 없음 — build_video_map.py 먼저 실행하면 p 키로 영상 재생 가능)")
        return
    total = sum(1 for _ in csv.DictReader(open(MAP_CSV, encoding='utf-8')))
    print(f"  영상 매핑: 대본 {total}개 중 {len(vmap)}개에 영상 연결됨")


def open_video(rel):
    vmap = load_video_map()
    video_path = vmap.get(rel)
    if not video_path:
        print("  [!] 대응 영상 없음 (video_map.csv에 매핑 없음).")
        return
    if not Path(video_path).exists():
        print(f"  [!] 매핑된 영상 파일이 실제로 없음: {video_path}")
        return
    sys_name = platform.system()
    try:
        if VIDEO_PLAYER == "vlc":
            vlc = VLC_PATHS.get(sys_name, "vlc")
            subprocess.Popen([vlc, f"--rate={VIDEO_RATE}", str(video_path)])
            print(f"  >> {Path(video_path).name} (VLC {VIDEO_RATE}x)")
        else:
            if sys_name == 'Darwin':
                subprocess.run(['open', str(video_path)])
            elif sys_name == 'Windows':
                subprocess.run(['start', '', str(video_path)], shell=True)
            else:
                subprocess.run(['xdg-open', str(video_path)])
            print(f"  >> {Path(video_path).name}")
    except FileNotFoundError:
        print(f"  [ERROR] VLC 못 찾음. 설치 확인 / VLC_PATHS 수정 / 또는 VIDEO_PLAYER='default'")
    except Exception as e:
        print(f"  [ERROR] {e}")


# ---- 진행 기록 ----
def load_progress():
    if not PROGRESS_CSV.exists():
        return set()
    with open(PROGRESS_CSV, newline='', encoding='utf-8') as f:
        return {row['file'] for row in csv.DictReader(f)}


def append_progress(summary):
    needs_header = not PROGRESS_CSV.exists() or PROGRESS_CSV.stat().st_size == 0
    with open(PROGRESS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(summary)


def reset_file(rel):
    cleaned_path = OUTPUT_DIR / rel
    if cleaned_path.exists():
        cleaned_path.unlink()
    if PROGRESS_CSV.exists():
        with open(PROGRESS_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or PROGRESS_FIELDS
            rows = [r for r in reader if r['file'] != rel]
        with open(PROGRESS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def display_words(words):
    for i, w in enumerate(words):
        text = w.get('text', '')
        start = w.get('start', 0)
        end = w.get('end', 0)
        wtype = w.get('type', 'word')
        display = "·" if wtype == 'spacing' else text
        marker = "  <- 외국어?" if (wtype == 'word' and is_likely_foreign(text)) else ""
        print(f"  [{i:3}] {display:18s} ({start:5.2f}-{end:5.2f}s){marker}")


def print_help():
    print("-" * 66)
    print("  [편집]")
    print("  4         ->  인덱스 4 삭제")
    print("  4-8       ->  인덱스 4~8 연속 삭제  (4-8o / 4-8b 도 가능)")
    print("  4o        ->  인덱스 4 overlap (음성 겹침: prosody 분석서 제외)")
    print("  4b        ->  인덱스 4 bad_text (오전사: 텍스트 분석서 제외)")
    print("  2,4-8o,10b -> 혼합 입력")
    print("  B         ->  파일 전체 텍스트 불량 (text_reliable=false)")
    print("  Enter     ->  변경 없이 저장하고 다음 파일")
    print("  [이동]")
    print("  g         ->  점프: 파일명 입력 -> 그 파일로 이동 + 초기화 + 재편집")
    print("  s         ->  skip: 저장 안 하고 다음 (미처리로 남아 다음에 다시 뜸)")
    print("  [기타]")
    print("  p         ->  play: 대응 영상을 VLC 1.5배속 재생")
    print("  ?         ->  help: 명령어 목록 다시 보기")
    print("  q         ->  quit: 저장하고 종료 (다음에 이어서)")
    print("-" * 66)


def parse_command(cmd, num_words):
    actions = {}
    for token in cmd.split(','):
        token = token.strip()
        if not token:
            continue
        if token[-1] in ('o', 'b'):
            suffix, body = token[-1], token[:-1]
        else:
            suffix, body = '', token
        action = {'': 'delete', 'o': 'overlap', 'b': 'bad_text'}[suffix]
        if '-' in body:
            parts = body.split('-')
            if len(parts) != 2:
                return None
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                return None
            if lo > hi:
                lo, hi = hi, lo
            idx_range = range(lo, hi + 1)
        else:
            try:
                idx_range = [int(body)]
            except ValueError:
                return None
        for idx in idx_range:
            if idx < 0 or idx >= num_words:
                return None
            actions[idx] = action
    return actions


def process_file(jf, rel):
    """반환: (result, payload). result in {done, skip, quit, goto}"""
    try:
        with open(jf, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] 읽기 실패: {e}")
        return 'skip', None

    words = data.get('words', [])
    text = data.get('text', '').strip()
    duration = data.get('audio_duration_secs', 0)
    lang = data.get('language_code', 'unknown')

    if lang != 'kor':
        print(f"  [!] 언어 감지: '{lang}' - 한국어 아님! (B로 전체 표시 고려)")
    if rel not in load_video_map():
        print(f"  [!] 대응 영상 없음 - p 안 됨. 텍스트만 처리하거나 s로 건너뛰기")
    print(f"  길이 {duration:.2f}s")
    print(f"  텍스트: \"{text}\"")
    display_words(words)

    foreign = [i for i, w in enumerate(words)
               if w.get('type') == 'word' and is_likely_foreign(w.get('text', ''))]
    if foreign:
        print(f"  외국어 의심 인덱스: {foreign}  (b로 표시)")

    file_bad = False
    actions = {}
    while True:
        cmd = input("  > ").strip()
        lc = cmd.lower()
        if lc == 'q':
            return 'quit', None
        if cmd == '?':
            print_help(); continue
        if lc == 'p':
            open_video(rel); continue
        if lc == 's':
            return 'skip', None
        if lc == 'g':
            target = input("  이동할 파일명 (일부만 입력 가능): ").strip()
            if not target:
                print("  취소."); continue
            return 'goto', target
        if cmd == 'B':
            file_bad = True; break
        if cmd == '':
            break
        parsed = parse_command(cmd, len(words))
        if parsed is None:
            print("  [!] 형식 오류. 예: '4' / '4-8' / '4o' / '7b' / '2,4-8o,10b' / 'B'")
            continue
        actions = parsed
        break

    delete_idx = {i for i, a in actions.items() if a == 'delete'}
    overlap_idx = {i for i, a in actions.items() if a == 'overlap'}
    bad_idx = {i for i, a in actions.items() if a == 'bad_text'}

    deleted_texts = []
    new_words = []
    for i, w in enumerate(words):
        if i in delete_idx:
            if w.get('type') == 'word':
                deleted_texts.append(w.get('text', ''))
            continue
        w = dict(w)
        if i in overlap_idx:
            w['flag'] = 'overlap'
        elif i in bad_idx:
            w['flag'] = 'bad_text'
        new_words.append(w)

    while new_words and new_words[0].get('type') == 'spacing':
        new_words.pop(0)
    while new_words and new_words[-1].get('type') == 'spacing':
        new_words.pop()
    merged = []
    for w in new_words:
        if merged and merged[-1].get('type') == 'spacing' and w.get('type') == 'spacing':
            m = dict(merged[-1]); m['end'] = w['end']; merged[-1] = m
            continue
        merged.append(w)
    new_words = merged

    text_tokens = [w for w in new_words if w.get('flag') != 'bad_text']
    new_text = ''.join(w.get('text', '') for w in text_tokens).strip()
    while '  ' in new_text:
        new_text = new_text.replace('  ', ' ')

    new_data = dict(data)
    new_data['words'] = new_words
    new_data['text'] = new_text
    new_data['text_reliable'] = not file_bad

    dst = OUTPUT_DIR / Path(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(smart_format(new_data))

    summary = {
        'file': rel,
        'deleted': len(delete_idx),
        'overlap': len(overlap_idx),
        'bad_text': len(bad_idx),
        'text_reliable': not file_bad,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    append_progress(summary)

    msgs = []
    if delete_idx: msgs.append(f"삭제 {deleted_texts}")
    if overlap_idx: msgs.append(f"overlap {sorted(overlap_idx)}")
    if bad_idx: msgs.append(f"bad_text {sorted(bad_idx)}")
    if file_bad: msgs.append("파일 전체 text_reliable=false")
    print("  OK " + (" | ".join(msgs) if msgs else "변경 없이 저장"))
    if not file_bad:
        print(f"     텍스트: \"{new_text}\"")
    return 'done', summary


def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] {DATA_DIR} 폴더 없음.")
        return
    all_files = sorted(DATA_DIR.rglob("*.json"))
    if not all_files:
        print(f"[ERROR] JSON 없음.")
        return
    rel_list = [str(f.relative_to(DATA_DIR)) for f in all_files]

    processed = load_progress()
    print(f"전체: {len(all_files)} | 처리됨: {len(processed)} | 남음: {len(all_files) - len(processed)}")
    video_summary()
    print()
    print_help()

    idx = 0
    while idx < len(all_files):
        rel = rel_list[idx]
        if rel in processed:
            idx += 1
            continue
        remaining = sum(1 for r in rel_list if r not in processed)
        print(f"\n[남은 {remaining}개] {rel}")
        result, payload = process_file(all_files[idx], rel)

        if result == 'quit':
            print("\n저장 후 종료.")
            break
        elif result == 'done':
            processed.add(rel)
            idx += 1
        elif result == 'skip':
            idx += 1
        elif result == 'goto':
            matches = [r for r in rel_list if payload in r]
            if not matches:
                print(f"  [!] '{payload}' 매칭 파일 없음.")
                continue
            if len(matches) == 1:
                target = matches[0]
            else:
                # 여러 개 매칭 -> 번호로 선택
                print(f"  '{payload}' 매칭 {len(matches)}개 — 번호 선택 (Enter=취소):")
                for i, m in enumerate(matches[:20]):
                    print(f"     [{i}] {m}")
                if len(matches) > 20:
                    print(f"     ... 외 {len(matches) - 20}개 (더 구체적으로 입력하면 좁혀짐)")
                sel = input("  번호: ").strip()
                if not sel.isdigit() or int(sel) >= len(matches[:20]):
                    print("  취소.")
                    continue
                target = matches[int(sel)]
            reset_file(target)
            processed.discard(target)
            idx = rel_list.index(target)
            print(f"  -> '{target}' 초기화 + 이동. 다시 편집하세요.")
            continue

    print(f"\n=== 현황 ===")
    print(f"처리됨: {len(processed)} / 전체 {len(all_files)}")
    print(f"출력: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
