# UMCG Full 대 Russian Roulette 3,000-update 분석 안내

이 파일은 ChatGPT Pro가 이 결과 묶음을 분석할 때 가장 먼저 읽어야 하는 안내문이다.

## 1. 가장 중요한 비교 원칙

공식 비교 대상은 다음 두 실험이다.

- Full 기준선: [`02_full_3000_matched`](02_full_3000_matched/)
- Russian Roulette(RR, 확률적으로 최대 문맥 길이를 선택하는 그래디언트 추정법): [`03_rr_3000`](03_rr_3000/)

두 실험은 모두 Distributed Data Parallel(DDP, 여러 GPU가 같은 모델을 나누어 학습하는 방식)의 graph splitting을 끈 동일한 실행 조건을 사용한다.

[`90_auxiliary_runs/01_full_3000_original`](90_auxiliary_runs/01_full_3000_original/)은 보조 Full 실행이다. 이름이 더 짧아 보이지만 공식 RR 비교 기준으로 사용하면 안 된다.

## 2. `essential`과 `appendix`

- `essential`: 해당 실험의 결론을 재현하거나 비교하는 데 먼저 필요한 자료
- `appendix`: checkpoint별 manifest, 개별 진단, 사전 점검처럼 결론을 정밀 감사할 때 읽는 상세자료

원시 결과 문서는 삭제하지 않았다. 분석 우선순위에 따라 위치만 구분했다.

## 3. 권장 읽기 순서

1. 전체 결론: [`00_overview/essential/FINAL_REPORT.md`](00_overview/essential/FINAL_REPORT.md)
2. 초기 calibration:
   - [`01_calibration/essential/calibration_350m_exact_64_32_32.json.report.json`](01_calibration/essential/calibration_350m_exact_64_32_32.json.report.json)
   - [`01_calibration/essential/calibration_350m_exact_64_32_32.json`](01_calibration/essential/calibration_350m_exact_64_32_32.json)
3. 장기학습 핵심 비교: [`04_joint_analysis/essential/long_run_comparison.json`](04_joint_analysis/essential/long_run_comparison.json)
4. 학습 중 schedule 안정성: [`04_joint_analysis/essential/schedule_drift_summary.json`](04_joint_analysis/essential/schedule_drift_summary.json)
5. Full과 RR의 가중치 이동 비교: [`04_joint_analysis/essential/weight_trajectory_comparison.json`](04_joint_analysis/essential/weight_trajectory_comparison.json)
6. checkpoint별 통합 진단: [`04_joint_analysis/essential/checkpoint_diagnostics_summary.json`](04_joint_analysis/essential/checkpoint_diagnostics_summary.json)
7. 수치를 직접 다시 계산할 때만 두 실험의 `essential/metrics.jsonl`을 읽는다.
8. 개별 checkpoint의 전체 통계가 필요할 때만 [`04_joint_analysis/appendix/checkpoint_diagnostics`](04_joint_analysis/appendix/checkpoint_diagnostics/)를 읽는다.

## 4. 디렉터리 구조

| 디렉터리 | 역할 | 분석 우선순위 |
|---|---|---|
| [`00_overview`](00_overview/) | 최종 종합 보고서와 무결성·검증 자료 | 가장 먼저 |
| [`01_calibration`](01_calibration/) | 초기 full-coordinate calibration과 작은 사전 점검 | 높음 |
| [`02_full_3000_matched`](02_full_3000_matched/) | 공식 Full 3,000-update 기준선 | 높음 |
| [`03_rr_3000`](03_rr_3000/) | 공식 RR 3,000-update 실험 | 높음 |
| [`04_joint_analysis`](04_joint_analysis/) | 두 실험의 직접 비교, checkpoint 진단, 표와 그래프 | 높음 |
| [`90_auxiliary_runs`](90_auxiliary_runs/) | 보조·사전 점검·실패 실행 및 구현 메모 | 필요한 경우에만 |

## 5. 핵심 표와 그래프

표:

- [`checkpoint_diagnostics_summary.csv`](04_joint_analysis/essential/tables/checkpoint_diagnostics_summary.csv)
- [`weight_trajectory_comparison.csv`](04_joint_analysis/essential/tables/weight_trajectory_comparison.csv)

그래프:

- [Validation loss](04_joint_analysis/essential/plots/validation_loss.svg)
- [효율비](04_joint_analysis/essential/plots/efficiency_ratio.svg)
- [문맥별 그래디언트 정렬](04_joint_analysis/essential/plots/level_alignment.svg)
- [Full correction second moment](04_joint_analysis/essential/plots/correction_second_moment_full.svg)
- [RR correction second moment](04_joint_analysis/essential/plots/correction_second_moment_rr.svg)

## 6. 원본과 링크에 대한 주의

`FINAL_REPORT.md`는 권위 원본을 그대로 보존한 복사본이므로 문서 안의 링크가 `/home/ubuntu/checkpoint/keunyoung/umcg/...` 절대경로를 가리킨다. 복사본만 분석할 때는 그 링크 대신 이 안내문의 상대경로를 사용한다.

이 묶음에는 대용량 모델 가중치, 원본 실행 로그, canonical FP32 checkpoint export가 포함되지 않는다. 필요할 경우 다음 권위 원본에서 확인한다.

`/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32`

## 7. 무결성과 정리 상태

- 최초 복사된 Markdown, JSON(JavaScript Object Notation), JSON Lines 파일: 75개
- 정리 과정에서 삭제된 최초 결과 문서: 0개
- 이동 전후 75개 문서의 SHA-256 내용 해시 집합: 일치
- 권위 원본에서 추가한 분석 자료:
  - CSV 표 2개
  - SVG(Scalable Vector Graphics, 확대해도 깨지지 않는 벡터 그림) 5개
  - 원본 `SHA256SUMS` 1개

정리 후 새로 작성한 이 안내문과 각 디렉터리의 `README.md`는 탐색용 메타데이터이며 실험 산출물이 아니다.
