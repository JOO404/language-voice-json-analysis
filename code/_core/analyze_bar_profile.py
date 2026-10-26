#!/usr/bin/env python3
"""
사람 단위 BAR 8차원 행동 프로필 (8b 단계)
=========================================
앞 단계들이 만든 발화 단위 데이터를 사람 단위로 묶어서
연구실의 BAR(Behavioral Anchor Rating) 8차원에 매핑되는
자동 행동 지표를 계산.

8차원 (Merrill-Reid Social Styles 기반):
  Directness, Warmth, Dominance, Responsiveness,
  Emotional Expressiveness, Conflict Avoidance,
  Sarcasm/Sharpness, Social Distance
+ Social Style 분류 (Amiable / Expressive / Driver / Analytical)

각 지표는 raw 값 + z-score 정규화된 1-5 점수로 출력.
인간 3-rater 평가의 보조 지표로 사용 가능 (상관 분석 등).

입력:
  data/cleaned/*.json   — 텍스트, 격식 어미
  data/emotion/*.json   — KOTE 44 감정 (한국어 라벨 사용)
  data/prosody/*.json   — F0/RMS (prosody_reliable만 사용)
  data/intent/*.json    — dialog act 분포

출력:
  data/personality/person_<id>.json  — 사람별 상세
  personality_profiles.csv            — 22명 비교 표

사용:
  python analyze_bar_profile.py
"""
import json
import csv
import re
import sys
import statistics
from pathlib import Path
from collections import defaultdict

CLEANED_DIR = Path("data/cleaned")
EMOTION_DIR = Path("data/emotion")
PROSODY_DIR = Path("data/prosody")
INTENT_DIR = Path("data/intent")
OUTPUT_DIR = Path("data/personality")
SUMMARY_CSV = Path("personality_profiles.csv")

# === KOTE 한국어 라벨 그룹 ===
WARMTH_EMOTIONS = ['행복', '기쁨', '즐거움/신남', '편안/쾌적', '아껴주는',
                   '고마움', '감동/감탄', '환영/호의', '안심/신뢰']
ANGER_EMOTIONS = ['화남/분노', '짜증', '증오/혐오']
SARCASM_EMOTIONS = ['어이없음', '한심함', '비장함', '우쭐댐/무시함']

# === 한국어 격식 어미 ===
FORMAL_ENDINGS = ['습니다', '습니까', '십니다', '입니다', '습니', '읍니']
POLITE_ENDINGS = ['아요', '어요', '에요', '예요', '죠']


def count_endings(text):
    """격식/비격식 어미 카운트"""
    formal = sum(text.count(e) for e in FORMAL_ENDINGS)
    polite = sum(text.count(e) for e in POLITE_ENDINGS)
    return formal, polite


def mean_or(values, default=None):
    v = [x for x in values if x is not None]
    return statistics.mean(v) if v else default


def std_or(values, default=None):
    v = [x for x in values if x is not None]
    return statistics.stdev(v) if len(v) >= 2 else default


def aggregate_person(person_id, clips):
    """clips: 한 사람의 모든 클립 dict 리스트 (cleaned/emotion/prosody/intent)"""
    word_counts = []
    formal_total = 0
    polite_total = 0
    intent_counts = defaultdict(int)
    intent_total = 0
    warmth_scores = []
    anger_scores = []
    sarcasm_scores = []
    top1_intensities = []
    f0_means_rel = []
    f0_within_clip_stds = []
    rms_means_rel = []
    speech_rates_rel = []
    n_rel_prosody = 0

    for c in clips:
        cl = c.get('cleaned')
        em = c.get('emotion')
        pr = c.get('prosody')
        it = c.get('intent')

        # 텍스트
        if cl and cl.get('text_reliable', True):
            text = (cl.get('text') or '').strip()
            if text:
                word_counts.append(len(text.split()))
                fo, po = count_endings(text)
                formal_total += fo
                polite_total += po

        # 의도
        if it and it.get('primary_intent'):
            intent_counts[it['primary_intent']] += 1
            intent_total += 1

        # 감정
        if em and em.get('all_scores'):
            scores = em['all_scores']
            warmth_scores.append(max((scores.get(e, 0) for e in WARMTH_EMOTIONS), default=0))
            anger_scores.append(max((scores.get(e, 0) for e in ANGER_EMOTIONS), default=0))
            sarcasm_scores.append(max((scores.get(e, 0) for e in SARCASM_EMOTIONS), default=0))
            if scores:
                top1_intensities.append(max(scores.values()))

        # 운율 (reliable만)
        if pr and pr.get('prosody_reliable'):
            fs = pr.get('file_summary') or {}
            if fs.get('f0_mean') is not None:
                f0_means_rel.append(fs['f0_mean'])
                n_rel_prosody += 1
                if fs.get('f0_std') is not None:
                    f0_within_clip_stds.append(fs['f0_std'])
                if fs.get('rms_mean') is not None:
                    rms_means_rel.append(fs['rms_mean'])
                if fs.get('speech_rate_wps') is not None:
                    speech_rates_rel.append(fs['speech_rate_wps'])

    # intent 비율
    intent_dist = {k: (intent_counts.get(k, 0) / intent_total if intent_total else 0)
                   for k in ['statement', 'question', 'affirmation', 'negation', 'hesitation']}

    raw = {
        'n_clips': len(clips),
        'n_with_text': len(word_counts),
        'n_with_intent': intent_total,
        'n_with_emotion': len(warmth_scores),
        'n_prosody_reliable': n_rel_prosody,

        'avg_utterance_length': mean_or(word_counts, 0),
        'formal_ratio': (formal_total / (formal_total + polite_total)
                         if (formal_total + polite_total) > 0 else 0),
        'intent_distribution': intent_dist,

        'warmth_emotion_mean': mean_or(warmth_scores, 0),
        'anger_emotion_mean': mean_or(anger_scores, 0),
        'sarcasm_emotion_mean': mean_or(sarcasm_scores, 0),
        'emotion_intensity_std': std_or(top1_intensities, 0),

        'f0_mean_overall': mean_or(f0_means_rel),
        'f0_within_clip_mean_std': mean_or(f0_within_clip_stds),
        'rms_mean': mean_or(rms_means_rel),
        'speech_rate_mean': mean_or(speech_rates_rel),
    }

    # ===== BAR 8차원 raw 신호 =====
    raw['bar_signals'] = {
        # 직설성: (긍정+부정) 분명한 stance 비율 - 망설임 비율
        'directness':
            intent_dist['affirmation'] + intent_dist['negation'] - intent_dist['hesitation'],

        # 따뜻함: 긍정 감정 강도
        'warmth': raw['warmth_emotion_mean'],

        # 주도성: 평균 발화 길이 × 평서 비율 (길고 단정적 = 주도적)
        'dominance':
            (raw['avg_utterance_length'] / 10.0) * intent_dist['statement'],

        # 반응성: 긍정/맞장구 비율
        'responsiveness': intent_dist['affirmation'],

        # 감정 표현성: 감정 강도 변동 + 운율(F0 변동) 결합
        'expressiveness':
            raw['emotion_intensity_std'] +
            ((raw['f0_within_clip_mean_std'] or 0) / 50.0),  # Hz를 ~0-1로 정규화

        # 갈등 회피: 망설임 비율 - 분노 감정 강도
        'conflict_avoidance':
            intent_dist['hesitation'] - raw['anger_emotion_mean'],

        # 비꼼/날카로움: KOTE의 비꼼류 감정 강도
        'sarcasm': raw['sarcasm_emotion_mean'],

        # 사회적 거리감: 격식 어미 비율 (높을수록 거리감)
        'social_distance': raw['formal_ratio'],
    }
    return raw


def normalize_dimensions(profiles):
    """각 차원을 22명에 대해 z-score → 1-5 점수로 매핑"""
    dim_names = ['directness', 'warmth', 'dominance', 'responsiveness',
                 'expressiveness', 'conflict_avoidance', 'sarcasm', 'social_distance']

    for d in dim_names:
        values = [p['bar_signals'].get(d) for p in profiles.values()]
        valid = [v for v in values if v is not None]
        if len(valid) < 2:
            for p in profiles.values():
                p.setdefault('bar_scores_1_5', {})[d] = 3.0
            continue
        mu = statistics.mean(valid)
        sigma = statistics.stdev(valid)
        if sigma == 0:
            for p in profiles.values():
                p.setdefault('bar_scores_1_5', {})[d] = 3.0
            continue
        for pid, p in profiles.items():
            v = p['bar_signals'].get(d)
            if v is None:
                p.setdefault('bar_scores_1_5', {})[d] = None
            else:
                z = (v - mu) / sigma
                # z=-2 → 1점, z=0 → 3점, z=+2 → 5점, [1,5] 클리핑
                score = max(1.0, min(5.0, 3.0 + z))
                p.setdefault('bar_scores_1_5', {})[d] = round(score, 2)

    return profiles


def classify_social_style(scores):
    """Merrill-Reid 4분면 분류"""
    assertive = (scores.get('dominance', 3) + scores.get('directness', 3)) / 2
    responsive = (scores.get('warmth', 3) + scores.get('expressiveness', 3)) / 2
    high_a = assertive >= 3.0
    high_r = responsive >= 3.0
    if high_a and high_r:
        return 'Expressive'
    if high_a and not high_r:
        return 'Driver'
    if not high_a and high_r:
        return 'Amiable'
    return 'Analytical'


def main():
    if not CLEANED_DIR.exists():
        print(f"[ERROR] {CLEANED_DIR} 없음")
        return 1

    print("발화별 데이터 인덱싱 중...")
    clips_by_person = defaultdict(list)
    n_files = 0
    for jf in sorted(CLEANED_DIR.rglob("*.json")):
        n_files += 1
        rel = jf.relative_to(CLEANED_DIR)
        m = re.search(r'\d+', jf.parent.name)
        if not m:
            continue
        pid = m.group()

        clip = {'cleaned': None, 'emotion': None, 'prosody': None, 'intent': None}
        try:
            with open(jf, encoding='utf-8') as f:
                clip['cleaned'] = json.load(f)
        except Exception:
            pass
        for d, key in [(EMOTION_DIR, 'emotion'),
                       (PROSODY_DIR, 'prosody'),
                       (INTENT_DIR, 'intent')]:
            p = d / rel
            if p.exists():
                try:
                    with open(p, encoding='utf-8') as f:
                        clip[key] = json.load(f)
                except Exception:
                    pass
        clips_by_person[pid].append(clip)

    print(f"  → 전체 {n_files}개 클립 / {len(clips_by_person)}명")

    # 집계
    print("\n사람별 행동 신호 집계 중...")
    profiles = {}
    for pid in sorted(clips_by_person.keys(), key=lambda x: (len(x), x)):
        profiles[pid] = aggregate_person(pid, clips_by_person[pid])
        print(f"  · 사람 {pid}: {profiles[pid]['n_clips']} 클립 "
              f"(텍스트 {profiles[pid]['n_with_text']}, "
              f"감정 {profiles[pid]['n_with_emotion']}, "
              f"의도 {profiles[pid]['n_with_intent']}, "
              f"운율 {profiles[pid]['n_prosody_reliable']})")

    # 정규화
    print("\nBAR 차원 z-score → 1-5점 변환 중...")
    profiles = normalize_dimensions(profiles)

    # Social Style 분류
    print("Social Style 분류 중...")
    for pid, p in profiles.items():
        p['social_style'] = classify_social_style(p['bar_scores_1_5'])

    # JSON 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pid, p in profiles.items():
        with open(OUTPUT_DIR / f"person_{pid}.json", 'w', encoding='utf-8') as f:
            json.dump({'person': pid, **p}, f, ensure_ascii=False, indent=2)

    # CSV 저장
    fieldnames = [
        'person', 'n_clips', 'n_reliable_prosody', 'social_style',
        'directness', 'warmth', 'dominance', 'responsiveness',
        'expressiveness', 'conflict_avoidance', 'sarcasm', 'social_distance',
        'avg_utterance_length', 'formal_ratio',
        'pct_statement', 'pct_question', 'pct_affirmation',
        'pct_negation', 'pct_hesitation',
    ]
    rows = []
    for pid in sorted(profiles.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        p = profiles[pid]
        s = p['bar_scores_1_5']
        i = p['intent_distribution']
        rows.append({
            'person': pid,
            'n_clips': p['n_clips'],
            'n_reliable_prosody': p['n_prosody_reliable'],
            'social_style': p['social_style'],
            'directness': s.get('directness'),
            'warmth': s.get('warmth'),
            'dominance': s.get('dominance'),
            'responsiveness': s.get('responsiveness'),
            'expressiveness': s.get('expressiveness'),
            'conflict_avoidance': s.get('conflict_avoidance'),
            'sarcasm': s.get('sarcasm'),
            'social_distance': s.get('social_distance'),
            'avg_utterance_length': round(p['avg_utterance_length'], 1),
            'formal_ratio': round(p['formal_ratio'], 3),
            'pct_statement': round(i['statement'] * 100, 1),
            'pct_question': round(i['question'] * 100, 1),
            'pct_affirmation': round(i['affirmation'] * 100, 1),
            'pct_negation': round(i['negation'] * 100, 1),
            'pct_hesitation': round(i['hesitation'] * 100, 1),
        })
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 콘솔 요약
    print()
    print("=" * 60)
    print("  사람별 BAR 행동 프로필 완성")
    print("=" * 60)
    print(f"\n총 {len(profiles)}명 분석")

    style_counts = defaultdict(int)
    for p in profiles.values():
        style_counts[p['social_style']] += 1
    print("\n[Social Style 분포]")
    for s in ['Amiable', 'Expressive', 'Driver', 'Analytical']:
        c = style_counts.get(s, 0)
        bar = '█' * c
        print(f"  {s:12s}: {c:2d}명  {bar}")

    print(f"\n사람별 JSON: {OUTPUT_DIR.resolve()}")
    print(f"비교 CSV : {SUMMARY_CSV.resolve()}")
    print(f"\n→ 인간 3-rater 평가와 자동 점수의 상관 분석에 사용 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
