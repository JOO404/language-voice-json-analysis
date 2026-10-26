#!/usr/bin/env python3
"""
의도 탐지 (8단계) — 규칙 기반
============================
각 발화의 화행(speech act)을 텍스트 패턴 + 운율 신호로 분류.

5개 라벨:
  question     의문문 (-까요?, 의문사, F0 상승)
  affirmation  긍정/동의 (네, 맞아요, 그렇죠)
  negation     부정 (안, 못, 아니, 없어요)
  hesitation   망설임 (어/음 짧은 발화, 글쎄, 잘 모르겠)
  statement    평서 (기본값)

입력:
  data/cleaned/*.json   — 텍스트 (bad_text 제외된 정제 텍스트)
  data/prosody/*.json   — 운율 (있고 prosody_reliable=true일 때만 F0 slope 사용)

출력:
  data/intent/<같은 상대경로>.json  — 파일별 의도 + 점수 + 신호
  intent_summary.csv                  — 한 행 = 한 파일

text_reliable=false 인 파일은 건너뜀.

사용:
  python detect_intent.py
  python detect_intent.py --limit 5
  python detect_intent.py --source formatted    # cleaned 부족할 때
"""
import json
import csv
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

CLEANED_DIR = Path("data/cleaned")
PROSODY_DIR = Path("data/prosody")
OUTPUT_DIR = Path("data/intent")
SUMMARY_CSV = Path("intent_summary.csv")

# ===================== 규칙 패턴 =====================

# 의문 종결어미 (?'를 떼고 비교 — 텍스트 끝에 오는지)
QUESTION_ENDINGS = (
    "까요", "까",
    "나요", "나",
    "ㄴ가요", "ㄴ가", "는가요", "는가", "인가요", "인가",
    "ㄹ까요", "ㄹ까", "을까요", "을까",
    "니까요", "니까",
    "는지요", "는지", "인지요", "인지",
    "죠", "지요",
)

# 의문사
QUESTION_WORDS = (
    "누구", "누가", "뭐", "무엇", "무슨", "어디", "언제",
    "왜", "어떻게", "어떤", "어느", "얼마", "몇",
)

# 긍정/동의 (시작에 오면 강력)
AFFIRMATION_TOKENS = (
    "네", "예", "응",
    "맞아", "맞아요", "맞다", "맞습니다",
    "그래요", "그래", "그렇죠", "그렇네요", "그렇지",
    "좋아요", "좋아", "괜찮아요", "괜찮아",
    "알겠어요", "알겠습니다", "이해했어요",
)

# 부정 표현 (전체 텍스트에서 검색)
NEGATION_TOKENS = (
    "아니에요", "아니요", "아니야", "아냐", "아니",
    "없어요", "없어", "없습니다", "없다", "없는",
    "모르겠어요", "모르겠습니다", "모르겠어", "몰라요", "몰라",
)

# 망설임 필러 (짧은 발화에서 점수)
HESITATION_FILLERS = ("어", "음", "그", "저", "에", "아", "으", "어어", "음음")

# 망설임 헷지 표현
HESITATION_PHRASES = (
    "글쎄", "잘 모르겠", "그게", "아마도", "아마",
    "어떻게 보면", "뭐랄까", "뭐라고 해야",
    "이렇게 보면", "음...", "어...",
)

# === 간접화법(보고화법) 표지 ===
# Part 4 (상황 발화 과제) / Part 5 (감정 발화) 처럼 직접 수행이 요구되는 부분에서
# "~라고 말할 것 같아요" 식의 보고형 응답을 식별하기 위함.
# 5개 의도(content)와 직교하는 차원(form).

# 강한 신호: 인용 표지(라고/다고) + 발화 동사 변형
REPORTED_SPEECH_PATTERNS = (
    # 직접인용 라고 + 발화동사
    "라고 말", "라고 대답", "라고 답", "라고 표현", "라고 인사",
    "라고 이야기", "라고 부탁", "라고 묻", "라고 부르",
    "라고 하겠", "라고 할",
    # 간접인용 다고 + 발화동사
    "다고 말", "다고 대답", "다고 답", "다고 표현", "다고 이야기",
    "다고 하겠", "다고 할",
    # 정도로 / 식으로 + 발화동사
    "정도로 말", "정도로 답", "정도로 표현", "정도로 이야기",
    "식으로 말", "식으로 답", "식으로 표현",
)

# 약한 신호: 발화 동사 + 가정/추측 어미
HYPOTHETICAL_SPEECH_ENDINGS = (
    "말할 것 같", "답할 것 같", "대답할 것 같", "표현할 것 같",
    "이야기할 것 같", "인사할 것 같", "부탁할 것 같",
    "말할 거예요", "답할 거예요", "대답할 거예요",
    "말하겠어요", "답하겠어요", "대답하겠어요",
    "말하지 않을까", "답하지 않을까", "대답하지 않을까",
)

# F0 slope (Hz/s) 임계값 — 마지막 단어 기울기
F0_RISE_THRESHOLD = 30   # 이상이면 상승 (의문 단서)
F0_FALL_THRESHOLD = -30  # 이하면 하강 (평서 단서)


def detect_speech_mode(text):
    """
    발화가 직접 수행(direct)인지 간접/보고형(reported)인지 분류.

    Part 4 (Situational Speech Task) / Part 5 (Emotion/Speech Style) 처럼
    "상대에게 말하듯이" 직접 발화가 요구되는 부분에서, 일부 참가자가
    "~라고 말할 것 같아요" 식의 보고화법으로 답한 경우 자동 식별.

    Returns: (mode, signals)
      mode: "direct" | "reported" | "unknown"
      signals: 검출된 표지 리스트
    """
    text = text.strip()
    if not text:
        return "unknown", []

    signals = []

    # 1) 강한 신호: 인용 표지 + 발화동사
    for pat in REPORTED_SPEECH_PATTERNS:
        if pat in text:
            signals.append(("reported", f"인용+발화동사: '{pat}…'"))
            return "reported", signals

    # 2) 약한 신호: 발화동사 + 가정 어미
    for ending in HYPOTHETICAL_SPEECH_ENDINGS:
        if ending in text:
            signals.append(("reported", f"가정형 발화: '…{ending}'"))
            return "reported", signals

    return "direct", signals


def detect_intent(text, prosody=None):
    """
    Returns: (primary, scores, signals, confidence)
    signals = [(intent_label, evidence_str), ...]
    """
    text = text.strip()
    if not text:
        return "unknown", {}, [], "low"

    scores = {"statement": 1.0, "question": 0.0, "affirmation": 0.0,
              "negation": 0.0, "hesitation": 0.0}
    signals = []

    words = text.split()
    text_stripped = text.rstrip(".!?,;").strip()

    # ---------- QUESTION ----------
    if text.rstrip().endswith("?"):
        scores["question"] += 3
        signals.append(("question", "물음표(?)"))
    # 종결어미 (가장 긴 매치 우선)
    for end in sorted(QUESTION_ENDINGS, key=len, reverse=True):
        if text_stripped.endswith(end):
            scores["question"] += 2
            signals.append(("question", f"의문 종결어미: -{end}"))
            break
    # 의문사
    for qw in QUESTION_WORDS:
        if qw in text:
            scores["question"] += 1
            signals.append(("question", f"의문사: {qw}"))
            break

    # ---------- AFFIRMATION ----------
    if words:
        first = words[0].rstrip(".,!?;")
        if first in AFFIRMATION_TOKENS:
            scores["affirmation"] += 2
            signals.append(("affirmation", f"시작 토큰: {first}"))
        # 전체 발화에서 명확 긍정 표현
        seen = False
        for tok in ("맞아요", "그렇죠", "맞아", "그래요", "맞습니다", "알겠어요"):
            if tok in text:
                if not seen:
                    scores["affirmation"] += 1
                    signals.append(("affirmation", f"긍정 표현: {tok}"))
                    seen = True

    # ---------- NEGATION ----------
    for neg in sorted(NEGATION_TOKENS, key=len, reverse=True):
        if neg in text:
            scores["negation"] += 2
            signals.append(("negation", f"부정 표현: {neg}"))
            break
    # "안 " or "못 " 부정 표지
    if re.search(r"(^|\s)안\s\S", text):
        scores["negation"] += 1
        signals.append(("negation", '부정 표지: "안 "'))
    if re.search(r"(^|\s)못\s\S", text):
        scores["negation"] += 1
        signals.append(("negation", '부정 표지: "못 "'))

    # ---------- HESITATION ----------
    if words and len(words) <= 4:
        clean_words = [w.rstrip(".,!?;") for w in words]
        filler_n = sum(1 for w in clean_words if w in HESITATION_FILLERS)
        if filler_n > 0 and filler_n / len(words) >= 0.5:
            scores["hesitation"] += 3
            signals.append(("hesitation",
                            f"짧은 필러 위주 발화 ({filler_n}/{len(words)})"))
    for phrase in HESITATION_PHRASES:
        if phrase in text:
            scores["hesitation"] += 2
            signals.append(("hesitation", f"헷지 표현: {phrase}"))
            break

    # ---------- PROSODY (있고 신뢰 가능할 때) ----------
    if prosody and prosody.get("prosody_reliable"):
        ws = prosody.get("words", [])
        if ws:
            last = ws[-1]
            slope = last.get("f0_slope")
            if slope is not None:
                if slope >= F0_RISE_THRESHOLD:
                    scores["question"] += 1
                    signals.append(("question",
                                    f"마지막 단어 F0 상승: +{slope:.0f} Hz/s"))
                elif slope <= F0_FALL_THRESHOLD:
                    scores["statement"] += 0.5
                    signals.append(("statement",
                                    f"마지막 단어 F0 하강: {slope:.0f} Hz/s"))

    # ---------- 최종 결정 ----------
    primary = max(scores, key=scores.get)
    max_score = scores[primary]
    # 기본값(statement=1)만 있고 다른 점수 없으면 confidence=low
    if max_score <= 1:
        primary = "statement"
        confidence = "low"
    elif max_score >= 3:
        confidence = "high"
    else:
        confidence = "medium"

    return primary, scores, signals, confidence


def main():
    ap = argparse.ArgumentParser(description="규칙 기반 의도 탐지")
    ap.add_argument("--source", default="cleaned", choices=["cleaned", "formatted"],
                    help="입력 폴더 (기본 cleaned)")
    ap.add_argument("--limit", type=int, default=None, help="처음 N개만 (시험)")
    args = ap.parse_args()

    source_dir = Path("data") / args.source
    if not source_dir.exists():
        print(f"[ERROR] 입력 폴더 없음: {source_dir}")
        return 1

    files = sorted(source_dir.rglob("*.json"))
    if args.limit:
        files = files[: args.limit]
        print(f"[시험 모드] 처음 {len(files)}개\n")

    print(f"입력: {source_dir} ({len(files)}개)")
    print(f"운율 보조: {PROSODY_DIR}  ({'있음' if PROSODY_DIR.exists() else '없음 — 텍스트만 사용'})")
    print(f"출력: {OUTPUT_DIR.resolve()}\n")

    summary_rows = []
    counts = defaultdict(int)
    confidence_counts = defaultdict(int)
    speech_mode_counts = defaultdict(int)
    ok = 0
    skipped = 0

    for i, jf in enumerate(files, 1):
        rel = jf.relative_to(source_dir)
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f"  [{i:4d}/{len(files)}] 처리 {ok}  건너뜀 {skipped}")

        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"    [!] 읽기 실패 {rel}: {e}")
            continue

        text = (data.get("text") or "").strip()
        text_reliable = data.get("text_reliable", True)

        # 건너뛰기
        if not text_reliable or not text:
            skipped += 1
            out = {
                "person": data.get("words", [{}])[0].get("person", "") if False else "",
                "text": text, "text_reliable": text_reliable,
                "skip_reason": "text_reliable=false" if not text_reliable else "텍스트 비어있음",
                "primary_intent": None,
            }
            out_path = OUTPUT_DIR / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            continue

        # 운율 가져오기 (있으면)
        prosody = None
        prosody_path = PROSODY_DIR / rel
        if prosody_path.exists():
            try:
                with open(prosody_path, encoding="utf-8") as f:
                    prosody = json.load(f)
            except Exception:
                prosody = None

        # 사람/코드 추출
        parent = jf.parent.name
        m = re.search(r"\d+", parent)
        person = m.group() if m else ""
        code = jf.stem

        # 의도 탐지
        primary, scores, signals, confidence = detect_intent(text, prosody)
        # 발화 모드 탐지 (직접 vs 보고)
        speech_mode, speech_signals = detect_speech_mode(text)

        ok += 1
        counts[primary] += 1
        confidence_counts[confidence] += 1
        speech_mode_counts[speech_mode] += 1

        out = {
            "person": person,
            "code": code,
            "text": text,
            "text_reliable": text_reliable,
            "primary_intent": primary,
            "confidence": confidence,
            "speech_mode": speech_mode,
            "scores": {k: round(v, 2) for k, v in scores.items()},
            "signals": [[intent, ev] for intent, ev in signals],
            "speech_mode_signals": [[m, s] for m, s in speech_signals],
            "used_prosody": prosody is not None and prosody.get("prosody_reliable", False),
        }
        out_path = OUTPUT_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        summary_rows.append({
            "file": str(rel),
            "person": person,
            "code": code,
            "text": text,
            "primary_intent": primary,
            "speech_mode": speech_mode,
            "confidence": confidence,
            "n_signals": len(signals),
            "used_prosody": out["used_prosody"],
        })

    # 요약 CSV
    if summary_rows:
        fieldnames = ["file", "person", "code", "text", "primary_intent",
                      "speech_mode", "confidence", "n_signals", "used_prosody"]
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)

    # 요약 출력
    print()
    print("-" * 60)
    print(f"전체 {len(files)}개 | 처리 {ok} | 건너뜀 {skipped}")
    if counts:
        print("\n[의도 분포]")
        total = sum(counts.values())
        for intent in ["statement", "question", "affirmation", "negation", "hesitation"]:
            c = counts.get(intent, 0)
            bar = "█" * int(c / total * 40)
            print(f"  {intent:12s} {c:4d}건 ({c/total*100:5.1f}%) {bar}")
    if confidence_counts:
        print("\n[신뢰도 분포]")
        for level in ["high", "medium", "low"]:
            c = confidence_counts.get(level, 0)
            print(f"  {level:6s}: {c}건")
    if speech_mode_counts:
        print("\n[발화 모드 분포]")
        total_sm = sum(speech_mode_counts.values())
        for mode in ["direct", "reported"]:
            c = speech_mode_counts.get(mode, 0)
            bar = "█" * int(c / total_sm * 40) if total_sm > 0 else ""
            print(f"  {mode:8s}: {c:4d}건 ({c/total_sm*100:5.1f}%) {bar}")
        print("  → reported는 \"~라고 말할 것 같아요\" 식 보고화법 응답.")
        print("     Part 4/5 직접 발화 요청에 묘사로 답한 경우 식별용.")
    print(f"\n파일별 결과: {OUTPUT_DIR.resolve()}")
    if summary_rows:
        print(f"요약 CSV: {SUMMARY_CSV.resolve()}  ({len(summary_rows)}행)")

    # 샘플
    if summary_rows:
        print("\n[샘플 결과 — 각 의도별 1개씩]")
        seen = set()
        for row in summary_rows:
            if row["primary_intent"] not in seen and row["confidence"] in ("high", "medium"):
                seen.add(row["primary_intent"])
                print(f"  [{row['primary_intent']:11s}] \"{row['text'][:50]}{'...' if len(row['text'])>50 else ''}\"")
                if len(seen) >= 5:
                    break

    return 0


if __name__ == "__main__":
    sys.exit(main())
