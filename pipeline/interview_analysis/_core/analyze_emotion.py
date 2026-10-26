#!/usr/bin/env python3
"""
한국어 감정 분류 (6단계) — KOTE
=================================
data/cleaned/*.json의 text를 KOTE 44감정 모델로 분류.

KOTE: searle-j/kote_for_easygoing_people
  - 한국어 일상/감정 텍스트로 fine-tuning된 KcELECTRA 기반 멀티라벨 분류기
  - 44개 세부 감정 (기쁨/슬픔/불안/짜증/감동/지긋지긋... 등)
  - 첫 실행 시 모델 ~500MB 다운로드 후 캐시됨

출력:
  data/emotion/<같은 상대경로>.json   — 파일별 상세 (top-k + 전체 점수)
  emotion_summary.csv                  — 전체 요약 표 (Excel 친화 BOM 포함)

text_reliable=false / 텍스트 비어있는 파일은 건너뜀.

사용:
  python analyze_emotion.py                       # 전체 (cleaned 기준)
  python analyze_emotion.py --limit 5             # 시험 (처음 5개)
  python analyze_emotion.py --source formatted    # cleaned 부족할 때
  python analyze_emotion.py --topk 3              # CSV에 상위 N (기본 5)
"""
import json
import csv
import time
import argparse
import sys
import re
from pathlib import Path

MODEL_ID = "searle-j/kote_for_easygoing_people"
OUTPUT_DIR = Path("data/emotion")
SUMMARY_CSV = Path("emotion_summary.csv")


def load_pipeline():
    """KOTE 파이프라인 로드 (첫 실행이면 모델 다운로드)."""
    print(f"모델 로딩: {MODEL_ID}")
    print("  (첫 실행이면 ~500MB 다운로드 후 캐시됨. 1-3분 소요)")
    t0 = time.time()
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TextClassificationPipeline,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    # KOTE는 멀티라벨 분류라 sigmoid가 필수 (softmax이면 점수가 다 작아짐)
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1,                    # CPU (-1), GPU는 0,1...
        top_k=None,                   # 모든 라벨 점수
        function_to_apply="sigmoid",  # ★ 멀티라벨이므로 sigmoid
        truncation=True,
    )
    print(f"  ✓ 로드 완료 ({time.time()-t0:.1f}초)\n")
    return pipe


def parse_file_key(rel_path):
    """경로에서 (사람번호, 클립코드) 추출
    transcribed_06_leechulho/A_01_1_1.json -> ('06', 'A_01_1_1')"""
    p = Path(rel_path)
    m = re.search(r'\d+', p.parent.name)
    person = m.group() if m else ''
    code = p.stem
    return person, code


def fmt_min(seconds):
    if seconds < 60:
        return f"{seconds:.0f}초"
    return f"{seconds/60:.1f}분"


def main():
    ap = argparse.ArgumentParser(description="한국어 감정 분류 (KOTE)")
    ap.add_argument("--source", default="cleaned", choices=["cleaned", "formatted"],
                    help="입력 폴더 (기본: cleaned)")
    ap.add_argument("--limit", type=int, default=None,
                    help="처음 N개만 처리 (시험용)")
    ap.add_argument("--topk", type=int, default=5,
                    help="요약 CSV의 상위 N 감정 (기본 5)")
    args = ap.parse_args()

    source_dir = Path("data") / args.source
    if not source_dir.exists():
        print(f"[ERROR] 입력 폴더 없음: {source_dir}")
        if args.source == "cleaned":
            print("  → annotate_words.py로 정제 작업이 시작됐는지 확인")
            print("  → 아직 정제 전이면: python analyze_emotion.py --source formatted")
        return 1

    files = sorted(source_dir.rglob("*.json"))
    if not files:
        print(f"[ERROR] {source_dir}에 JSON 없음.")
        return 1

    if args.limit:
        files = files[: args.limit]
        print(f"[시험 모드] 처음 {len(files)}개만 처리\n")

    print(f"입력: {source_dir}  ({len(files)}개 JSON)")
    print(f"출력: {OUTPUT_DIR.resolve()}")
    if args.source == "formatted":
        print("  ⚠️ formatted 사용 — 인터뷰어 발화 등 미정제 텍스트 포함됨")
    print()

    pipe = load_pipeline()

    summary_rows = []
    ok = 0
    skipped = 0
    failed = 0
    start = time.time()

    for i, jf in enumerate(files, 1):
        rel = jf.relative_to(source_dir)

        # 진행 표시
        if i == 1 or i % 50 == 0 or i == len(files):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(files) - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{len(files)}] 처리 {ok}  건너뜀 {skipped}  실패 {failed}"
                  f"   ({rate:.1f}건/초, ETA {fmt_min(eta)})")

        try:
            with open(jf, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            failed += 1
            continue

        text = (data.get('text') or '').strip()
        text_reliable = data.get('text_reliable', True)  # formatted엔 없는 필드, 기본 True
        person, code = parse_file_key(str(rel))

        out = {
            "person": person,
            "code": code,
            "text": text,
            "text_reliable": text_reliable,
        }

        # 건너뛰기 조건
        skip_reason = None
        if not text_reliable:
            skip_reason = "text_reliable=false"
        elif not text:
            skip_reason = "텍스트 비어있음"

        if skip_reason:
            skipped += 1
            out["skip_reason"] = skip_reason
            out["top_emotions"] = []
            out["all_scores"] = {}
        else:
            try:
                result = pipe(text)[0]   # [{label, score}, ...] (44개)
                sorted_scores = sorted(result, key=lambda x: x['score'], reverse=True)
                top = sorted_scores[: args.topk]
                out["top_emotions"] = [
                    {"label": r['label'], "score": round(r['score'], 4)} for r in top
                ]
                out["all_scores"] = {
                    r['label']: round(r['score'], 4) for r in sorted_scores
                }
                ok += 1

                row = {"file": str(rel), "person": person, "code": code, "text": text}
                for k in range(args.topk):
                    if k < len(top):
                        row[f"top{k+1}_label"] = top[k]['label']
                        row[f"top{k+1}_score"] = round(top[k]['score'], 4)
                    else:
                        row[f"top{k+1}_label"] = ""
                        row[f"top{k+1}_score"] = ""
                summary_rows.append(row)
            except Exception as e:
                failed += 1
                out["error"] = str(e)[:200]
                out["top_emotions"] = []
                out["all_scores"] = {}

        # 파일별 결과 저장
        out_path = OUTPUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # 요약 CSV (BOM 포함 — Excel에서 한글 깨짐 방지)
    if summary_rows:
        fieldnames = ["file", "person", "code", "text"]
        for k in range(args.topk):
            fieldnames += [f"top{k+1}_label", f"top{k+1}_score"]
        with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)

    elapsed = time.time() - start
    print()
    print("-" * 60)
    print(f"전체 {len(files)}개 | 처리 {ok} | 건너뜀 {skipped} | 실패 {failed}")
    print(f"소요 시간: {fmt_min(elapsed)}")
    print(f"파일별 결과: {OUTPUT_DIR.resolve()}")
    if summary_rows:
        print(f"요약 표: {SUMMARY_CSV.resolve()}  ({len(summary_rows)}행)")

    # 첫 결과 미리보기
    if ok > 0:
        print("\n[샘플 결과]")
        first = next((f for f in OUTPUT_DIR.rglob("*.json")), None)
        if first:
            with open(first, encoding='utf-8') as f:
                sample = json.load(f)
            if sample.get('top_emotions'):
                print(f"  파일: {first.relative_to(OUTPUT_DIR)}")
                print(f"  텍스트: \"{sample['text'][:60]}{'...' if len(sample['text'])>60 else ''}\"")
                print("  상위 감정:")
                for e in sample['top_emotions'][:5]:
                    print(f"    · {e['label']:14s} {e['score']:.3f}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
