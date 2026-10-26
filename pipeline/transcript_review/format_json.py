#!/usr/bin/env python3
"""
JSON 일괄 포맷팅 도구
=====================
지저분하게 한 줄로 압축된 JSON 파일들을 사람이 읽기 좋게 들여쓰기.

특징:
- `words`처럼 동질적인 배열은 각 항목을 '한 줄에' 표시 (가독성)
- 그 외 구조는 표준 indent로 들여쓰기
- 원본은 건드리지 않고 새 폴더에 저장 (덮어쓰기 옵션도 있음)

사용법:
    python format_json.py                    # data/raw → data/formatted
    python format_json.py --in-place         # 원본 덮어쓰기 (주의!)
    python format_json.py --src foo --dst bar
"""

import json
import argparse
from pathlib import Path


# 한 줄로 압축할 배열 키 이름들 (필요시 추가)
COMPACT_ARRAY_KEYS = {"words"}


def smart_format(data, indent=2):
    """
    재귀적으로 JSON을 포맷.
    COMPACT_ARRAY_KEYS에 있는 키의 배열은 각 항목을 한 줄로.
    """

    def fmt(val, depth=0, parent_key=None):
        pad = " " * (indent * depth)
        inner_pad = " " * (indent * (depth + 1))

        # 1) compact array 처리 (예: "words")
        if (
            isinstance(val, list)
            and parent_key in COMPACT_ARRAY_KEYS
            and val
        ):
            parts = [
                f"{inner_pad}{json.dumps(item, ensure_ascii=False)}"
                for item in val
            ]
            return "[\n" + ",\n".join(parts) + f"\n{pad}]"

        # 2) 일반 dict
        if isinstance(val, dict):
            if not val:
                return "{}"
            parts = []
            for k, v in val.items():
                k_json = json.dumps(k, ensure_ascii=False)
                v_str = fmt(v, depth + 1, parent_key=k)
                parts.append(f"{inner_pad}{k_json}: {v_str}")
            return "{\n" + ",\n".join(parts) + f"\n{pad}}}"

        # 3) 일반 list (compact가 아닌 경우)
        if isinstance(val, list):
            if not val:
                return "[]"
            parts = [
                f"{inner_pad}{fmt(item, depth + 1)}" for item in val
            ]
            return "[\n" + ",\n".join(parts) + f"\n{pad}]"

        # 4) 그 외 primitive 값
        return json.dumps(val, ensure_ascii=False)

    return fmt(data, 0)


def process_file(src_path, dst_path):
    """파일 하나 포맷팅"""
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)
    formatted = smart_format(data, indent=2)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(formatted)


def main():
    parser = argparse.ArgumentParser(description="JSON 일괄 포맷팅")
    parser.add_argument("--src", default="data/raw", help="원본 폴더")
    parser.add_argument("--dst", default="data/formatted", help="출력 폴더")
    parser.add_argument("--in-place", action="store_true",
                        help="원본 덮어쓰기 (--src를 그대로 출력 폴더로 사용)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = src_dir if args.in_place else Path(args.dst)

    if not src_dir.exists():
        print(f"[ERROR] 소스 폴더 없음: {src_dir}")
        return

    # rglob: 하위 폴더까지 재귀적으로 모든 JSON 찾기
    json_files = sorted(src_dir.rglob("*.json"))
    if not json_files:
        print(f"[ERROR] {src_dir} 및 하위 폴더에 JSON 파일이 없습니다.")
        return

    # 폴더별 통계 (몇 명/몇 개 폴더에서 왔는지 보여줌)
    folders = sorted({jf.parent for jf in json_files})
    print(f"검색된 폴더: {len(folders)}개")
    print(f"검색된 JSON: {len(json_files)}개")

    if args.in_place:
        ans = input(f"⚠️  {len(json_files)}개 파일을 원본 덮어쓸 거야. 계속? [y/N]: ")
        if ans.strip().lower() != "y":
            print("취소.")
            return

    ok, fail = 0, 0
    for jf in json_files:
        # 원본 폴더 기준 상대 경로 계산 → 출력 폴더에도 동일한 구조로
        rel_path = jf.relative_to(src_dir)
        dst = dst_dir / rel_path
        try:
            process_file(jf, dst)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {jf}: {e}")
            fail += 1

    print(f"\n완료: 성공 {ok}개, 실패 {fail}개")
    print(f"출력 위치: {dst_dir.resolve()}")


if __name__ == "__main__":
    main()
