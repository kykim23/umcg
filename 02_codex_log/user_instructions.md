# 사용자 지시사항

최종 갱신: 2026-07-23 (Asia/Seoul)

## 상시 지시

- 모든 답변과 문서는 가독성을 우선한다. 축약어가 필요하면 처음 사용할 때 원뜻을 설명한다.
- 이 파일, `conversation_history.md`, `work_log.md`를 매 작업 시작 전에 확인하고 작업 중 최신 상태로 유지한다.
- 구현 또는 실험에서 사용자 결정이 필요한 중요한 선택은 임의로 확정하지 않고 질문한다.
- 실험 중 GPU를 점유하는 `galore` Conda 환경 프로세스는 종료해도 된다. 이 권한은 실제 실험이 진행되는 동안에만 적용한다.
- 본격적인 작업 전 서버 하드웨어와 현재 점유 상태를 확인한다.
- 새 Conda 환경을 만들지 않는다. 기존 `umcg` 환경에 필요한 패키지를 설치한다.
- 로컬 C4 원본은 `/home/ubuntu/data/c4_en/en`을 사용한다.
- 원본 log, 결과, estimator 설정, calibration 보고서는 `/home/ubuntu/checkpoint/keunyoung/umcg` 아래에 보존한다.
- 모든 실험이 끝날 때마다 생성된 Markdown(`.md`), JSON(`.json`), JSON Lines(`.jsonl`) 결과 문서를 `/home/ubuntu/keunyoung/workspace/umcg/results/<experiment_id>/`에도 동일하게 복사한다. 원본 상대 디렉터리 구조를 유지하고, 파일 수·크기·SHA-256을 대조해 복사본을 검증한다. `/home/ubuntu/checkpoint/keunyoung/umcg`의 원본은 계속 권위 있는 장기 보존본으로 유지한다.
- 저장소의 `results`는 ChatGPT Pro가 실험결과를 읽고 분석하는 입력 경로다. Primary 결과와 auxiliary·preflight·failed 자료가 혼동되지 않도록 구분하고, 분석 시작 파일과 canonical 비교 대상을 안내하는 휴대 가능한 index를 제공한다. 결과 복사본의 삭제·이동 범위에 판단이 필요하면 먼저 사용자 승인을 받는다.
- ChatGPT Pro용 `results/<experiment_id>/` 복사본은 각 주요 실험을 `essential`(필수자료)과 `appendix`(상세자료·부록)로 나눈다. 공식 비교 실험과 auxiliary·preflight·failed 실행은 별도 디렉터리로 구분하고, 루트 `ANALYSIS_INDEX.md`에서 권장 읽기 순서와 canonical 비교 대상을 안내한다.
- H100 검증은 bfloat16(BF16)만 수행하고 FP16 실험은 생략한다.
- 검증 gate가 실패하면 원인을 수정한 뒤 실패한 gate부터 재개한다. 이미 통과한 CPU 검사를 처음부터 반복하지 않는다.

## 확정된 구현 정책

- 학습 parser는 `src/umcg/cli/arguments.py`를 단일 원본으로 유지한다.
- `training/runner.py` 시작부에서 지원 RuntimeConfig 인자를 기능별로 즉시 확인할 수 있게 한다.
- `--estimator_config`와 `--gradient_estimator`의 역할 차이를 CLI help와 문서에 명확히 적는다.
- Russian Roulette 일정 `[1.0, 0.5, 0.25, 0.125]`는 별도의 미검증 smoke 설정으로만 둔다. Q=1 안전 템플릿은 유지한다.
- 최종 calibration은 측정 64 / 선택 32 / 독립 감사 32 parent batch로 분리한다.
- `workers=1`을 유지하며 이번 작업에서는 실제 병렬 tokenization을 구현하지 않는다.
- 문서 종료 토큰(End of Sequence, EOS)은 현행 정책을 유지한다. 정확한 최대 문맥 배수 길이에서 EOS 단독 조각을 버리는 동작을 문서화하고 경계 테스트를 추가한다.
- CountSketch를 과학적 calibration 판정 경로에서 제거하고 full-size gradient의 streaming Gram 통계를 사용한다.
- Calibration은 H100 두 장에서 논리 parent batch 128, rank당 physical batch 상한 64로 수행한다. Exact 경로의 VRAM 상한은 85%이며 64가 실패하면 physical 32를 두 chunk로 합쳐 논리 batch 128을 유지한다.
- Calibration 자료 규모는 논리 batch 기준 측정 64 / 선택 32 / 감사 32이며, 실제 parent sample은 각각 8,192 / 4,096 / 4,096개다.
- Tail probability 후보는 `Q_512=1`과 `{1, 0.75, 0.5, 0.25, 0.125}`에서 단조감소를 만족하는 35개만 평가한다.
- Calibration timing은 `timing_repeats=1`로 수행한다.
- 초기 독립 감사를 통과한 schedule은 학습 중 변경하지 않고 checkpoint 재측정은 drift 진단에만 사용한다.
- 이번 비교 실험은 같은 초기 weight로 Full 3,000 update와 Russian Roulette 3,000 update를 두 H100에서 순차 실행한다. Warm-up은 300 update다.
