#!/usr/bin/env python3
"""
운율 분석 결과 품질 점검 (QC)
=============================
data/prosody/*.json 의 분석 결과를 통계적으로 점검하여
잡음/분석 실패가 의심되는 파일을 자동으로 골라냄.

점검 항목:
  1. F0 검출률: 단어 중 유효한 F0 값을 가진 비율 (낮으면 PYIN이 잡음에 막힘)
  2. 유성음 비율: voiced_ratio 평균 (낮으면 비주기적 잡음 비중 큼)
  3. F0 합리성: 평균 F0가 정상 음성 범위(80-300 Hz) 안인지
  4. 화자 일관성: 같은 사람의 다른 클립들과 평균 F0가 비슷한지

각 파일을 healthy / suspect / failed 로 자동 분류.

출력:
  prosody_qc_report.csv  — 파일별 QC 결과 (의심 순 정렬)
  prosody_qc_summary.txt — 전체 통계 요약

사용:
  python qc_prosody.py
"""
import json
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROSODY_DIR = Path("data/prosody")
QC_CSV = Path("prosody_qc_report.csv")
QC_SUMMARY = Path("prosody_qc_summary.txt")

# === 합리성 기준 ===
F0_DETECT_THRESHOLD = 0.5    # 단어의 50% 이상에서 F0 검출되어야 함
VOICED_RATIO_THRESHOLD = 0.3  # 평균 voiced_ratio 0.3 이상이어야 함
NORMAL_F0_MIN = 80            # 정상 음성 F0 평균 하한 (Hz)
NORMAL_F0_MAX = 300           # 정상 음성 F0 평균 상한 (Hz)
PERSON_DIFF_THRESHOLD = 40    # 같은 사람 중앙값과의 차이 (Hz)


def main():
    if not PROSODY_DIR.exists():
        print(f"[ERROR] {PROSODY_DIR} 없음. analyze_prosody.py 먼저 실행하세요.")
        return 1

    files_data = []
    person_f0s = defaultdict(list)

    # 1차 통과: 통계 수집
    for jf in sorted(PROSODY_DIR.rglob("*.json")):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [!] 읽기 실패 {jf}: {e}")
            continue

        rel = str(jf.relative_to(PROSODY_DIR))
        person = d.get('person', '')
        code = d.get('code', '')

        # 분석 실패 (skip_reason 있음)
        if d.get('skip_reason'):
            files_data.append({
                "file": rel, "person": person, "code": code,
                "status": "failed",
                "n_words": 0, "n_with_f0": 0,
                "f0_detection_rate": None,
                "mean_voiced_ratio": None,
                "f0_mean": None, "f0_vs_person_med": None,
                "rms_mean": None,
                "issues": d['skip_reason'],
            })
            continue

        words = d.get('words', [])
        if not words:
            files_data.append({
                "file": rel, "person": person, "code": code,
                "status": "failed",
                "n_words": 0, "n_with_f0": 0,
                "f0_detection_rate": None,
                "mean_voiced_ratio": None,
                "f0_mean": None, "f0_vs_person_med": None,
                "rms_mean": None,
                "issues": "단어 없음",
            })
            continue

        n_words = len(words)
        n_with_f0 = sum(1 for w in words if w.get('f0_mean') is not None)
        f0_detection_rate = n_with_f0 / n_words

        voiced_ratios = [w.get('voiced_ratio', 0) for w in words]
        mean_vr = statistics.mean(voiced_ratios) if voiced_ratios else 0

        fs = d.get('file_summary', {})
        f0_mean = fs.get('f0_mean')
        rms_mean = fs.get('rms_mean')

        issues = []
        if f0_detection_rate < F0_DETECT_THRESHOLD:
            issues.append(f"F0검출률낮음({f0_detection_rate*100:.0f}%)")
        if mean_vr < VOICED_RATIO_THRESHOLD:
            issues.append(f"유성음비율낮음({mean_vr:.2f})")
        if f0_mean is not None:
            if f0_mean < NORMAL_F0_MIN:
                issues.append(f"F0비정상저음({f0_mean:.0f}Hz)")
            elif f0_mean > NORMAL_F0_MAX:
                issues.append(f"F0비정상고음({f0_mean:.0f}Hz)")
            person_f0s[person].append(f0_mean)

        files_data.append({
            "file": rel, "person": person, "code": code,
            "status": None,           # 2차 통과에서 결정
            "n_words": n_words,
            "n_with_f0": n_with_f0,
            "f0_detection_rate": round(f0_detection_rate, 3),
            "mean_voiced_ratio": round(mean_vr, 3),
            "f0_mean": f0_mean,
            "f0_vs_person_med": None,  # 2차 통과에서 계산
            "rms_mean": rms_mean,
            "issues": "; ".join(issues),
            "_issues_list": issues,
        })

    # 2차 통과: 화자별 일관성
    person_median = {p: statistics.median(fs) for p, fs in person_f0s.items() if fs}
    for row in files_data:
        if row['status'] == "failed":
            continue
        f0 = row['f0_mean']
        p = row['person']
        if f0 is not None and p in person_median:
            diff = abs(f0 - person_median[p])
            row['f0_vs_person_med'] = round(diff, 1)
            if diff > PERSON_DIFF_THRESHOLD:
                row['_issues_list'].append(f"화자평균과큰차이({diff:.0f}Hz)")

        row['issues'] = "; ".join(row['_issues_list'])
        row['status'] = "healthy" if not row['_issues_list'] else "suspect"
        del row['_issues_list']

    # 정렬: failed → suspect (이슈 많은 순) → healthy
    status_order = {"failed": 0, "suspect": 1, "healthy": 2}
    files_data.sort(key=lambda r: (
        status_order.get(r['status'], 9),
        -len(r['issues']) if r.get('issues') else 0,
        r['file']
    ))

    # CSV 저장
    fieldnames = ["file", "person", "code", "status", "n_words", "n_with_f0",
                  "f0_detection_rate", "mean_voiced_ratio",
                  "f0_mean", "f0_vs_person_med", "rms_mean", "issues"]
    with open(QC_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(files_data)

    # 요약
    counts = defaultdict(int)
    for r in files_data:
        counts[r['status']] += 1

    lines = []
    lines.append("=" * 60)
    lines.append("  운율 분석 품질 점검(QC) 요약")
    lines.append("=" * 60)
    lines.append(f"\n전체 파일: {len(files_data)}")
    lines.append(f"  ✓ healthy (정상): {counts['healthy']}  ({counts['healthy']/len(files_data)*100:.1f}%)")
    lines.append(f"  ⚠ suspect (의심): {counts['suspect']}  ({counts['suspect']/len(files_data)*100:.1f}%)")
    lines.append(f"  ✗ failed  (실패): {counts['failed']}  ({counts['failed']/len(files_data)*100:.1f}%)")

    lines.append(f"\n--- 사람별 F0 통계 (median, n={len(person_median)}명) ---")
    for p in sorted(person_median.keys()):
        f0s = person_f0s[p]
        lines.append(f"  {p:5s}: 중앙값 {person_median[p]:6.1f} Hz   "
                     f"범위 {min(f0s):.0f}–{max(f0s):.0f}   (n={len(f0s)} 클립)")

    issue_counter = defaultdict(int)
    for r in files_data:
        if r['issues']:
            for issue in r['issues'].split('; '):
                key = issue.split('(')[0]
                issue_counter[key] += 1
    if issue_counter:
        lines.append("\n--- 의심 사유별 빈도 ---")
        for issue, c in sorted(issue_counter.items(), key=lambda x: -x[1]):
            lines.append(f"  {issue:25s}: {c}건")

    # 상위 의심 5개
    suspects = [r for r in files_data if r['status'] == 'suspect'][:5]
    if suspects:
        lines.append(f"\n--- 가장 의심스러운 파일 Top 5 (자세히 들어볼 후보) ---")
        for r in suspects:
            f0 = f"{r['f0_mean']}Hz" if r['f0_mean'] is not None else "F0없음"
            lines.append(f"  · {r['file']}")
            lines.append(f"      {f0}, 검출률 {r['f0_detection_rate']}, 사유: {r['issues']}")

    text = "\n".join(lines)
    QC_SUMMARY.write_text(text, encoding='utf-8')
    print(text)
    print(f"\n전체 CSV: {QC_CSV.resolve()}  (suspect가 맨 위에 정렬됨)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
