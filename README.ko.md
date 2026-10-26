# 인터뷰·음성 분석 코드

한국어 인터뷰 전사·검수, 언어·음성 분석, 보이스 샘플 비교에 사용한 최종 코드입니다.

[English](README.md) · [설치 및 전체 실행 순서](docs/RUN_GUIDE_EN.txt)

## 작업별 구성

| 작업 | 코드 폴더 | 포함 기능 | 산출물 |
| --- | --- | --- | --- |
| **1. 전사 준비·BAD/겹침 검수** | [transcript_review](pipeline/transcript_review) | 음성 추출, 전사, 단어 시간 정렬, 음성·영상 재생, BAD/겹침 표시, 시간 구간 검사 | 검수 JSON, 음성 측정 대상 JSON |
| **2. 인터뷰 언어·음성 분석** | [interview_analysis](pipeline/interview_analysis) | librosa 음성 특징, KOTE 텍스트 감정, 규칙 기반 의도, 참가자별 BAR 지표 집계 | 클립별 JSON, 요약 CSV |
| **3. 보이스 샘플 분석·비교** | [voice_samples](pipeline/voice_samples) | 같은 음성 분석법으로 후보 목소리 측정, 클립·목소리별 비교, 대사 일치·시간 민감도 확인 | 클립별 JSON, 목소리 비교 CSV |

**실행은 `run.py` 하나에서 작업을 선택합니다.** 각 폴더의 `_core`는 내부 계산용 코드이며 따로 실행하는 단계가 아닙니다. 보이스 샘플은 인터뷰와 같은 음성 계산 코드를 사용합니다.

```console
python run.py review --help
python run.py interview --help
python run.py voices --help
```

## BAD·겹침 검수

```console
python run.py review listen --project project
```

Windows 터미널에서 명령을 입력한 뒤 Enter를 누릅니다.

| 입력 | 동작 |
| --- | --- |
| Enter | 저장하고 다음 답변 자동 재생 |
| `p` | 다시 듣기 |
| `4b` / `4o` | 4번 단어 전사 오류(BAD) / 겹침 표시 |
| `B` | 답변 전체 전사문 불량 표시 |
| `v` | 연결된 영상 열기 |
| `q` | 종료 |

음성 재생과 영상 열기는 **1번 검수 작업에 포함**됩니다. 영상 연결에는 `project/video_map.csv`를 사용하며, 형식은 실행 안내에 있습니다.

기존 단어별 JSON이 있으면 검사·검수부터 시작합니다. 영상만 새로 받은 경우에는 음성 추출 → 전사 → 시간 정렬 → 검수 순서로 진행합니다. 클립 경계와 질문 번호는 입력으로 지정합니다.

## 실행 준비

Python 3.12를 사용했으며 전사 환경과 분석 환경을 구분합니다. `requirements/`는 패키지 버전, `config/`는 모델 정보·측정 설정·BAR 기준값입니다. 전체 명령은 [실행 안내](docs/RUN_GUIDE_EN.txt)에 정리되어 있습니다.

`tests/`와 `example/`에는 계산 검증과 가상 입력 예제가 있습니다. 원본 영상·음성, 실제 전사문, 참가자 명단·개별 결과 및 모델 파일은 별도 자료입니다.

KOTE는 텍스트 감정 분석이며, BAR는 보조 행동 지표입니다. 보이스 샘플은 그룹마다 대사가 다르므로 같은 그룹·질문 안에서 비교합니다.
