#!/usr/bin/env python3
"""
분석 결과 한국어 메타 정보 → 영어 일괄 변환
=============================================
이미 만들어진 출력 파일들의 한국어 메타 정보를 영어로 변환.
분석 결과의 영문 메타데이터를 생성합니다.

원본 한국어판은 보존, 영문판은 별도로 생성:
  prosody_qc_report.csv   → prosody_qc_report_en.csv
  data/intent/*.json      → data/intent_en/*.json
  data/prosody/*.json     → data/prosody_en/*.json

(인터뷰 발화 텍스트 자체와 한국어 어미/표지 인용은 그대로 유지 — 그건
번역하면 의미가 사라지는 raw 데이터 또는 morphological markers임)

사용:
  python translate_outputs_to_en.py
"""
import json
import csv
import re
import sys
from pathlib import Path


# ============ 번역 사전 ============

# QC 사유 (prosody_qc_report.csv의 issues, prosody JSON의 qc_issues)
QC_ISSUE_PREFIXES = {
    "F0검출률낮음": "low F0 detection rate",
    "유성음비율낮음": "low voiced ratio",
    "F0비정상저음": "abnormally low F0",
    "F0비정상고음": "abnormally high F0",
    "화자평균과큰차이": "deviation from speaker median",
}

# skip_reason (prosody/intent JSON)
SKIP_REASONS = {
    "분석 가능한 단어 없음 (overlap 제외 후 0개)": "no analyzable words (0 after overlap exclusion)",
    "분석 가능한 단어 없음": "no analyzable words",
    "WAV 없음": "WAV file missing",
    "WAV 로드 실패": "WAV load failed",
    "WAV 비어있음": "WAV is empty",
    "librosa 분석 실패": "librosa analysis failed",
    "유효한 단어 구간 없음": "no valid word segments",
    "단어 없음": "no word tokens",
    "텍스트 비어있음": "empty text",
    "text_reliable=false": "text_reliable=false",  # already English-like
    "WAV 로드 실패: ": "WAV load failed: ",
}

# Intent signals (data/intent/*.json의 signals + speech_mode_signals)
# 한국어 어미·표지가 뒤에 붙어있어서 prefix 매칭으로 처리
INTENT_SIGNAL_PREFIXES = {
    "물음표(?)": "question mark (?)",
    "의문 종결어미: -": "interrogative ending: -",
    "의문사: ": "question word: ",
    "시작 토큰: ": "starting token: ",
    "긍정 표현: ": "affirmation expression: ",
    "짧은 발화의 긍정 표현: ": "affirmation in short utterance: ",
    "부정 표현: ": "negation expression: ",
    "부정 표지: ": "negation marker: ",
    "짧은 필러 위주 발화 (": "filler-dominated short utterance (",
    "헷지 표현: ": "hedge phrase: ",
    "마지막 단어 F0 상승: +": "last-word F0 rising: +",
    "마지막 단어 F0 상승: ": "last-word F0 rising: ",
    "마지막 단어 F0 하강: ": "last-word F0 falling: ",
    "인용+발화동사: '": "quotative + speech verb: '",
    "가정형 발화: '": "hypothetical speech form: '",
}


# ============ 번역 함수 ============

def translate_qc_issue(text):
    """QC 사유 문자열 번역. 'F0검출률낮음(40%); ...' 형태."""
    if not text:
        return text
    parts = text.split('; ')
    out = []
    for part in parts:
        translated = part
        # QC issue prefix 매칭
        for kr in sorted(QC_ISSUE_PREFIXES.keys(), key=len, reverse=True):
            if part.startswith(kr):
                translated = QC_ISSUE_PREFIXES[kr] + part[len(kr):]
                break
        else:
            # skip reason 매칭
            for kr in sorted(SKIP_REASONS.keys(), key=len, reverse=True):
                if part.startswith(kr):
                    translated = SKIP_REASONS[kr] + part[len(kr):]
                    break
        # 영문 컨벤션: 영문자/숫자 뒤 ( 앞에 공백
        translated = re.sub(r'([a-zA-Z0-9])\(', r'\1 (', translated)
        out.append(translated)
    return '; '.join(out)


def translate_intent_signal(text):
    """Intent signal description 번역. 한국어 인용 부분은 그대로 유지."""
    if not text:
        return text
    for kr in sorted(INTENT_SIGNAL_PREFIXES.keys(), key=len, reverse=True):
        if text.startswith(kr):
            return INTENT_SIGNAL_PREFIXES[kr] + text[len(kr):]
    return text


def translate_skip_reason(text):
    """skip_reason 번역."""
    if not text:
        return text
    for kr in sorted(SKIP_REASONS.keys(), key=len, reverse=True):
        if text.startswith(kr):
            return SKIP_REASONS[kr] + text[len(kr):]
    return text


# ============ 변환 작업 ============

def translate_prosody_qc_csv():
    src = Path("prosody_qc_report.csv")
    if not src.exists():
        print(f"  · {src} 없음, 건너뜀")
        return False
    dst = Path("prosody_qc_report_en.csv")
    with open(src, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row.get('issues'):
            row['issues'] = translate_qc_issue(row['issues'])
    with open(dst, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ {src} → {dst}  ({len(rows)}행)")
    return True


def translate_prosody_jsons():
    src_dir = Path("data/prosody")
    dst_dir = Path("data/prosody_en")
    if not src_dir.exists():
        print(f"  · {src_dir} 없음, 건너뜀")
        return False
    count = 0
    for jf in src_dir.rglob("*.json"):
        with open(jf, encoding='utf-8') as f:
            data = json.load(f)
        # qc_issues
        if data.get('qc_issues'):
            data['qc_issues_kr'] = data['qc_issues']
            data['qc_issues'] = translate_qc_issue(data['qc_issues'])
        # skip_reason
        if data.get('skip_reason'):
            data['skip_reason_kr'] = data['skip_reason']
            data['skip_reason'] = translate_skip_reason(data['skip_reason'])
        # 저장
        rel = jf.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        count += 1
    print(f"  ✓ {src_dir}/ → {dst_dir}/  ({count}개)")
    return True


def translate_intent_jsons():
    src_dir = Path("data/intent")
    dst_dir = Path("data/intent_en")
    if not src_dir.exists():
        print(f"  · {src_dir} 없음, 건너뜀")
        return False
    count = 0
    for jf in src_dir.rglob("*.json"):
        with open(jf, encoding='utf-8') as f:
            data = json.load(f)
        # signals (의도 신호)
        if data.get('signals'):
            data['signals_kr'] = data['signals']
            data['signals'] = [[s[0], translate_intent_signal(s[1])] for s in data['signals']]
        # speech_mode_signals (발화 모드 신호)
        if data.get('speech_mode_signals'):
            data['speech_mode_signals_kr'] = data['speech_mode_signals']
            data['speech_mode_signals'] = [[s[0], translate_intent_signal(s[1])] for s in data['speech_mode_signals']]
        # skip_reason
        if data.get('skip_reason'):
            data['skip_reason_kr'] = data['skip_reason']
            data['skip_reason'] = translate_skip_reason(data['skip_reason'])
        # 저장
        rel = jf.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        count += 1
    print(f"  ✓ {src_dir}/ → {dst_dir}/  ({count}개)")
    return True


def main():
    print("=" * 60)
    print("  분석 출력의 한국어 메타 정보 → 영어 일괄 변환")
    print("=" * 60)
    print()

    print("[1/3] Prosody QC report (CSV)")
    translate_prosody_qc_csv()
    print()

    print("[2/3] Prosody JSON 파일들 (qc_issues, skip_reason)")
    translate_prosody_jsons()
    print()

    print("[3/3] Intent JSON 파일들 (signals, speech_mode_signals)")
    translate_intent_jsons()
    print()

    print("=" * 60)
    print("  완료")
    print("=" * 60)
    print()
    print("영문 산출물:")
    print()
    print("📊 CSV 표 (Excel에서 한글 안 깨짐)")
    print("    personality_profiles.csv       BAR 점수 + Social Style (전부 영어)")
    print("    emotion_summary_en.csv         KOTE 44 감정 영어 라벨")
    print("    prosody_file_summary.csv       운율 요약 (숫자만)")
    print("    prosody_qc_report_en.csv       ← NEW (QC 사유 영어)")
    print("    intent_summary.csv             의도/발화모드 (영어 컬럼)")
    print()
    print("📁 JSON 상세 (필요시 깊이 보기용)")
    print("    data/emotion_en/*.json         영어 KOTE 라벨")
    print("    data/prosody_en/*.json         ← NEW (영어 QC 메타)")
    print("    data/intent_en/*.json          ← NEW (영어 신호 설명)")
    print("    data/personality/*.json        BAR 사람별 프로필 (영어)")
    print()
    print("⚠ 본질적으로 한국어 (인터뷰 화자 발화 자체 → 번역 불가):")
    print("    intent_summary.csv 의 text 컬럼")
    print("    prosody_word_summary.csv 의 word_text 컬럼")
    print("    data/cleaned/*.json 의 text 필드")
    print("    영문 signal 안의 인용된 한국어 어미·표지 (예: 'starting token: 네')")
    print("    한국어 어미와 표지는 분석 근거이므로 원문을 유지합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
