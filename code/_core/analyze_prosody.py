#!/usr/bin/env python3
"""
운율 분석 (7단계) — librosa
============================
data/cleaned/*.json 의 word 토큰 타임스탬프를 사용해
data/wav/ 의 대응 WAV에서 단어별 F0(음높이) / RMS(세기) / 속도를 측정.

설계 결정 (정제 단계 표시와 자동 연동):
  - flag == "overlap"  : 운율 분석 제외 (음성 오염)
  - flag == "bad_text" : 포함 (음성은 멀쩡)
  - type == "spacing"  : 무시 (단어 사이 공백)
  - text_reliable=false: 텍스트만 못 믿는 거라 운율은 정상 분석

출력:
  data/prosody/<같은 상대경로>.json   — 파일별 상세 (단어별 측정값 + 파일 요약)
  prosody_file_summary.csv             — 한 행 = 한 파일
  prosody_word_summary.csv             — 한 행 = 한 단어 (대량 분석용)

사용:
  python analyze_prosody.py                       # 전체
  python analyze_prosody.py --limit 5             # 시험
  python analyze_prosody.py --source formatted    # cleaned 부족할 때
  python analyze_prosody.py --fmin 80 --fmax 300  # F0 탐색 범위 조정
"""
import json
import csv
import re
import time
import sys
import argparse
from pathlib import Path
import numpy as np

WAV_DIR = Path("data/wav")
OUTPUT_DIR = Path("data/prosody")
FILE_CSV = Path("prosody_file_summary.csv")
WORD_CSV = Path("prosody_word_summary.csv")

SR = 16000              # WAV 샘플레이트 (16kHz mono)
DEFAULT_FMIN = 65       # 성인 남성 저음까지 포함
DEFAULT_FMAX = 350      # 성인 여성 고음까지 포함
FRAME_LENGTH = 2048     # ~128 ms 윈도우
HOP_LENGTH = 512        # ~32 ms 간격


def fmt_min(seconds):
    if seconds < 60:
        return f"{seconds:.0f}초"
    return f"{seconds/60:.1f}분"


def parse_key(rel_path):
    p = Path(rel_path)
    m = re.search(r'\d+', p.parent.name)
    return (m.group() if m else ''), p.stem


def analyze_word_segment(f0, voiced_flag, rms, frame_times, start_t, end_t):
    """프레임 배열에서 [start_t, end_t] 구간만 잘라 통계 계산."""
    mask = (frame_times >= start_t) & (frame_times < end_t)
    w_f0 = f0[mask]
    w_voiced = voiced_flag[mask]
    w_rms = rms[mask]
    w_times = frame_times[mask]

    f0_valid_mask = ~np.isnan(w_f0)
    f0_valid = w_f0[f0_valid_mask]
    times_valid = w_times[f0_valid_mask]

    out = {
        "f0_mean": None, "f0_std": None, "f0_range": None, "f0_slope": None,
        "rms_mean": 0.0, "rms_max": 0.0,
        "voiced_ratio": 0.0,
        "n_voiced_frames": int(len(f0_valid)),
    }

    if len(f0_valid) >= 2:
        out["f0_mean"] = round(float(np.mean(f0_valid)), 2)
        out["f0_std"] = round(float(np.std(f0_valid)), 2)
        out["f0_range"] = round(float(np.max(f0_valid) - np.min(f0_valid)), 2)
        if len(f0_valid) >= 3:
            try:
                slope, _ = np.polyfit(times_valid, f0_valid, 1)
                out["f0_slope"] = round(float(slope), 2)
            except Exception:
                pass
    elif len(f0_valid) == 1:
        out["f0_mean"] = round(float(f0_valid[0]), 2)
        out["f0_std"] = 0.0
        out["f0_range"] = 0.0

    if len(w_voiced) > 0:
        out["voiced_ratio"] = round(float(np.mean(w_voiced)), 3)
    if len(w_rms) > 0:
        out["rms_mean"] = round(float(np.mean(w_rms)), 5)
        out["rms_max"] = round(float(np.max(w_rms)), 5)

    return out


def analyze_file(json_path, wav_path, fmin, fmax, librosa_mod):
    """한 파일의 운율 분석. (output_dict, error_str) 반환."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    all_words = data.get('words', [])
    # word 토큰만, overlap 제외
    word_tokens = [w for w in all_words if w.get('type') == 'word']
    words_to_analyze = [w for w in word_tokens if w.get('flag') != 'overlap']
    excluded_overlap = len(word_tokens) - len(words_to_analyze)

    if not words_to_analyze:
        return None, "분석 가능한 단어 없음 (overlap 제외 후 0개)"

    if not wav_path.exists():
        return None, f"WAV 없음: {wav_path.name}"

    # 오디오 로드
    try:
        y, _ = librosa_mod.load(str(wav_path), sr=SR, mono=True)
    except Exception as e:
        return None, f"WAV 로드 실패: {e}"

    if len(y) == 0:
        return None, "WAV 비어있음"

    # 전체에 대해 한 번 PYIN + RMS 계산 (단어별로 반복 호출하지 않음 — 훨씬 빠름)
    try:
        f0, voiced_flag, _ = librosa_mod.pyin(
            y, fmin=fmin, fmax=fmax, sr=SR,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )
        rms = librosa_mod.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
    except Exception as e:
        return None, f"librosa 분석 실패: {e}"

    voiced_flag = voiced_flag.astype(float)  # bool -> float for mean
    frame_times = librosa_mod.frames_to_time(
        np.arange(len(f0)), sr=SR, hop_length=HOP_LENGTH
    )

    # 단어별
    word_results = []
    for i, w in enumerate(words_to_analyze):
        start_t = float(w.get('start', 0))
        end_t = float(w.get('end', 0))
        if end_t <= start_t:
            continue
        stats = analyze_word_segment(f0, voiced_flag, rms, frame_times, start_t, end_t)
        word_results.append({
            "index": i,
            "text": w.get('text', ''),
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "duration": round(end_t - start_t, 3),
            **stats,
        })

    if not word_results:
        return None, "유효한 단어 구간 없음"

    # 파일 단위 요약
    durations = [w['duration'] for w in word_results]
    speech_duration = sum(durations)
    f0_means = [w['f0_mean'] for w in word_results if w['f0_mean'] is not None]
    rms_means = [w['rms_mean'] for w in word_results]

    person, code = parse_key(str(json_path))

    file_summary = {
        "num_words_analyzed": len(word_results),
        "num_words_excluded_overlap": excluded_overlap,
        "audio_duration_secs": data.get('audio_duration_secs', 0),
        "speech_duration": round(speech_duration, 3),
        "speech_rate_wps": round(len(word_results) / speech_duration, 3) if speech_duration > 0 else 0,
        "f0_mean": round(float(np.mean(f0_means)), 2) if f0_means else None,
        "f0_std": round(float(np.std(f0_means)), 2) if len(f0_means) >= 2 else None,
        "rms_mean": round(float(np.mean(rms_means)), 5) if rms_means else None,
    }

    output = {
        "person": person,
        "code": code,
        "text": data.get('text', ''),
        "text_reliable": data.get('text_reliable', True),
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "file_summary": file_summary,
        "words": word_results,
    }
    return output, None


def main():
    ap = argparse.ArgumentParser(description="운율 분석 (librosa)")
    ap.add_argument("--source", default="cleaned", choices=["cleaned", "formatted"],
                    help="입력 폴더 (기본 cleaned)")
    ap.add_argument("--limit", type=int, default=None, help="처음 N개만 (시험)")
    ap.add_argument("--fmin", type=int, default=DEFAULT_FMIN, help=f"F0 하한 Hz (기본 {DEFAULT_FMIN})")
    ap.add_argument("--fmax", type=int, default=DEFAULT_FMAX, help=f"F0 상한 Hz (기본 {DEFAULT_FMAX})")
    ap.add_argument("--resume", action="store_true",
                    help="이미 처리된 파일 건너뛰기 (이어하기)")
    ap.add_argument("--only-flagged", action="store_true",
                    help="prosody_qc_report.csv에서 suspect/failed로 표시된 파일만 재분석 "
                         "(--fmin/--fmax와 함께 strict 범위로 재시도하는 용도)")
    args = ap.parse_args()

    source_dir = Path("data") / args.source
    if not source_dir.exists():
        print(f"[ERROR] 입력 폴더 없음: {source_dir}")
        return 1
    if not WAV_DIR.exists():
        print(f"[ERROR] WAV 폴더 없음: {WAV_DIR}  (extract_wav.py 먼저 실행)")
        return 1

    print(f"입력 JSON: {source_dir}")
    print(f"입력 WAV : {WAV_DIR}")
    print(f"출력     : {OUTPUT_DIR.resolve()}")
    print(f"F0 탐색 범위: {args.fmin}–{args.fmax} Hz")
    if args.resume:
        print("[이어하기] 이미 처리된 파일은 건너뜀")
    print()

    # librosa는 무거우니 첫 호출 직전에 로드
    print("librosa 로딩...")
    t0 = time.time()
    import librosa
    print(f"  ✓ {time.time()-t0:.1f}초\n")

    files = sorted(source_dir.rglob("*.json"))
    if args.limit:
        files = files[: args.limit]
        print(f"[시험 모드] 처음 {len(files)}개\n")

    # --only-flagged: QC 리포트에서 suspect/failed 만 추림
    if args.only_flagged:
        qc_csv = Path("prosody_qc_report.csv")
        if not qc_csv.exists():
            print(f"[ERROR] --only-flagged 사용하려면 {qc_csv} 필요. "
                  f"'python qc_prosody.py' 먼저 실행")
            return 1
        flagged = set()
        with open(qc_csv, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] != 'healthy':
                    flagged.add(row['file'].replace('\\', '/'))
        # 파일 리스트를 flagged만으로 필터
        before = len(files)
        files = [f for f in files
                 if str(f.relative_to(source_dir)).replace('\\', '/') in flagged]
        print(f"[flagged-only 모드] {before} → {len(files)}개 재분석 대상\n")
        if not files:
            print("재분석할 파일이 없습니다 (모두 healthy).")
            return 0

    ok = 0
    skipped = 0
    failed = 0
    file_summary_rows = []
    word_summary_rows = []
    start = time.time()

    for i, jf in enumerate(files, 1):
        rel = jf.relative_to(source_dir)
        wav_path = WAV_DIR / rel.with_suffix(".wav")
        out_path = OUTPUT_DIR / rel

        # 이어하기 (--only-flagged는 강제 재분석이므로 resume 무시)
        if args.resume and not args.only_flagged and out_path.exists() and out_path.stat().st_size > 0:
            ok += 1
            continue

        # 진행 표시 (PYIN이 느려서 자주)
        if i == 1 or i % 25 == 0 or i == len(files):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(files) - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{len(files)}] 처리 {ok}  건너뜀 {skipped}  실패 {failed}"
                  f"   ({rate:.2f}건/초, ETA {fmt_min(eta)})")

        result, err = analyze_file(jf, wav_path, args.fmin, args.fmax, librosa)

        if result is None:
            skipped += 1
            # 건너뜀 정보도 저장
            person, code = parse_key(str(rel))
            placeholder = {
                "person": person, "code": code,
                "skip_reason": err,
                "file_summary": None, "words": [],
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(placeholder, f, ensure_ascii=False, indent=2)
            continue

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            ok += 1

            # CSV용 요약 행
            fs = result['file_summary']
            file_summary_rows.append({
                "file": str(rel),
                "person": result['person'],
                "code": result['code'],
                "n_words": fs['num_words_analyzed'],
                "excluded_overlap": fs['num_words_excluded_overlap'],
                "speech_duration": fs['speech_duration'],
                "speech_rate_wps": fs['speech_rate_wps'],
                "f0_mean": fs['f0_mean'],
                "f0_std": fs['f0_std'],
                "rms_mean": fs['rms_mean'],
            })
            for w in result['words']:
                word_summary_rows.append({
                    "file": str(rel),
                    "person": result['person'],
                    "code": result['code'],
                    "word_idx": w['index'],
                    "word_text": w['text'],
                    "start": w['start'],
                    "end": w['end'],
                    "duration": w['duration'],
                    "f0_mean": w['f0_mean'],
                    "f0_std": w['f0_std'],
                    "f0_range": w['f0_range'],
                    "f0_slope": w['f0_slope'],
                    "rms_mean": w['rms_mean'],
                    "rms_max": w['rms_max'],
                    "voiced_ratio": w['voiced_ratio'],
                    "n_voiced_frames": w['n_voiced_frames'],
                })
        except Exception as e:
            failed += 1
            print(f"    [!] 저장 실패 {rel}: {e}")

    # CSV 쓰기
    if file_summary_rows:
        with open(FILE_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(file_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(file_summary_rows)
    if word_summary_rows:
        with open(WORD_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(word_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(word_summary_rows)

    elapsed = time.time() - start
    print()
    print("-" * 60)
    print(f"전체 {len(files)}개 | 처리 {ok} | 건너뜀 {skipped} | 실패 {failed}")
    print(f"소요 시간: {fmt_min(elapsed)}")
    print(f"파일별 결과: {OUTPUT_DIR.resolve()}")
    if file_summary_rows:
        print(f"파일 요약 CSV: {FILE_CSV.resolve()}  ({len(file_summary_rows)}행)")
    if word_summary_rows:
        print(f"단어 요약 CSV: {WORD_CSV.resolve()}  ({len(word_summary_rows)}행)")

    # 샘플 미리보기
    if file_summary_rows:
        print("\n[샘플 결과] (첫 파일)")
        s = file_summary_rows[0]
        print(f"  파일: {s['file']}")
        print(f"  단어 {s['n_words']}개 (overlap 제외 {s['excluded_overlap']}개)")
        print(f"  발화 속도: {s['speech_rate_wps']} 단어/초")
        print(f"  평균 F0: {s['f0_mean']} Hz  (std {s['f0_std']})")
        print(f"  평균 세기: {s['rms_mean']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
