#!/usr/bin/env python3
"""
KOTE 한국어 라벨 → 영어 라벨 일괄 변환
=====================================
이미 만들어진 data/emotion/*.json 과 emotion_summary.csv 를
공식 영어 라벨로 변환합니다.

매핑은 KOTE 원논문 (arxiv 2205.05300) Appendix A "Emotion Clusters" 의
공식 영어 표기를 그대로 사용.

출력:
  data/emotion_en/<같은 상대경로>.json   — 영어 라벨 JSON (원본은 보존)
  emotion_summary_en.csv                   — 영어 라벨 CSV (원본은 보존)

사용:
  python translate_emotion_labels.py
"""
import json
import csv
import sys
from pathlib import Path

# === KOTE 원논문 Appendix A 공식 영어 표기 (44개) ===
# 출처: Jeon et al. (2022) "User Guide for KOTE", arXiv:2205.05300, Appendix A Table 4
KOTE_KR_TO_EN = {
    # Negative (25)
    '불평/불만':            'dissatisfaction',
    '당황/난처':            'embarrassment',
    '짜증':                'irritation',
    '슬픔':                'sadness',
    '절망':                'despair',
    '부끄러움':             'shame',
    '재미없음':             'boredom',
    '안타까움/실망':         'disappointment',
    '역겨움/징그러움':       'disgust',
    '경악':                'shock',
    '부담/안_내킴':         'reluctant',
    '부담/안 내킴':         'reluctant',          # 공백 변형 대응
    '공포/무서움':           'fear',
    '증오/혐오':            'contempt',
    '죄책감':              'guilt',
    '불안/걱정':            'anxiety',
    '의심/불신':            'distrust',
    '화남/분노':            'anger',
    '패배/자기혐오':         'self-loathing',      # 논문엔 "gessepany"로 적혀있으나 오기로 보임
    '귀찮음':              'laziness',
    '서러움':              'sorrow',
    '지긋지긋':            'fed up',
    '어이없음':             'preposterous',
    '불쌍함/연민':           'compassion',
    '한심함':              'pathetic',
    '힘듦/지침':            'exhaustion',
    # Positive (14)
    '감동/감탄':            'admiration',
    '행복':                'happiness',
    '기쁨':                'joy',
    '고마움':              'gratitude',
    '즐거움/신남':           'excitement',
    '아껴주는':             'care',
    '기대감':              'expectancy',
    '편안/쾌적':            'comfort',
    '환영/호의':            'welcome',
    '신기함/관심':           'interest',
    '안심/신뢰':            'relief',
    '존경':                'respect',
    '흐뭇함(귀여움/예쁨)':    'attracted',
    '뿌듯함':              'pride',
    # Neutral (4)
    '우쭐댐/무시함':         'arrogance',
    '놀람':                'surprise',
    '깨달음':              'realization',
    '비장함':              'resolute',
    # No emotion
    '없음':                'NO EMOTION',
}

INPUT_DIR = Path("data/emotion")
OUTPUT_DIR = Path("data/emotion_en")
INPUT_CSV = Path("emotion_summary.csv")
OUTPUT_CSV = Path("emotion_summary_en.csv")


def translate(label):
    """라벨이 매핑에 있으면 영어로, 없으면 원본 그대로 반환."""
    return KOTE_KR_TO_EN.get(label, label)


def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] {INPUT_DIR} 없음. analyze_emotion.py 먼저 실행하세요.")
        return 1

    unmapped = set()
    json_files = sorted(INPUT_DIR.rglob("*.json"))
    print(f"JSON 변환 시작: {len(json_files)}개")

    converted_json = 0
    for jf in json_files:
        try:
            with open(jf, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [!] 읽기 실패 {jf}: {e}")
            continue

        # top_emotions 리스트 변환 (한국어 원본도 label_kr 로 보존)
        for e in data.get('top_emotions') or []:
            kr = e.get('label', '')
            en = translate(kr)
            if en == kr and kr:
                unmapped.add(kr)
            e['label_kr'] = kr
            e['label'] = en

        # all_scores 딕셔너리의 키 변환
        if data.get('all_scores'):
            new_scores = {}
            for kr, score in data['all_scores'].items():
                en = translate(kr)
                if en == kr and kr:
                    unmapped.add(kr)
                new_scores[en] = score
            data['all_scores'] = new_scores

        # 저장
        out = OUTPUT_DIR / jf.relative_to(INPUT_DIR)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        converted_json += 1

    print(f"  ✓ {converted_json}개 변환 → {OUTPUT_DIR.resolve()}")

    # CSV 변환
    if INPUT_CSV.exists():
        with open(INPUT_CSV, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        for row in rows:
            for k, v in list(row.items()):
                if k.endswith('_label') and v:
                    if v in KOTE_KR_TO_EN:
                        row[k] = KOTE_KR_TO_EN[v]
                    elif v.strip():
                        unmapped.add(v)
        with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  ✓ CSV 변환 → {OUTPUT_CSV.resolve()}  ({len(rows)}행)")
    else:
        print(f"  · {INPUT_CSV} 없음 — CSV 변환 건너뜀")

    print()
    if unmapped:
        print(f"[경고] 매핑에 없는 라벨 {len(unmapped)}개 발견:")
        for l in sorted(unmapped):
            print(f"  · {l!r}")
        print("  → 변환하지 않고 원본 그대로 두었습니다. KOTE 매핑에 추가가 필요할 수 있습니다.")
    else:
        print("✓ 모든 라벨이 정상적으로 변환되었습니다.")

    # 샘플 미리보기
    if converted_json > 0:
        sample = next(OUTPUT_DIR.rglob("*.json"), None)
        if sample:
            with open(sample, encoding='utf-8') as f:
                d = json.load(f)
            if d.get('top_emotions'):
                print(f"\n[샘플 결과] {sample.relative_to(OUTPUT_DIR)}")
                print(f"  텍스트: \"{d.get('text','')[:50]}...\"")
                print("  상위 감정 (영어):")
                for e in d['top_emotions'][:5]:
                    print(f"    · {e['label']:18s} {e['score']:.3f}    ({e.get('label_kr','')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
