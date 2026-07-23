# 대화 이력

최종 갱신: 2026-07-23 (Asia/Seoul)

## 세션 요약

1. 사용자는 UMCG 저장소를 복구한 뒤 구조, 목적, 실험 준비 상태와 검증 순서를 요청했다.
2. 서버는 H100 80GB 두 장과 다수 CPU core를 갖춘 AI 연구 서버로 확인했다. 로컬 C4 원본은 네트워크 파일 시스템의 `/home/ubuntu/data/c4_en/en`에 있다.
3. Conda 환경은 새로 만들지 않고 `umcg`를 사용하며, 필요한 의존성은 그 환경에 설치하기로 했다.
4. H100 주 실험은 BF16만 수행하고 FP16은 생략하기로 했다.
5. EOS 정확 배수 정책은 현행을 유지하되, 마지막 token에서 EOS로 가는 target이 없다는 사실과 희소성이 미측정 가정임을 문서화하기로 했다.
6. 사용자는 학습 인자 parser 흐름, estimator 설정과 estimator 종류의 차이, `eval_parent_batches`, `local`/`local_raw`, `workers`, calibration을 설명하고 가독성을 개선해 달라고 요청했다.
7. 코드 감사 결과 Russian Roulette tail probability, 최대-level 확률, 하위 correction 누적, 한 최대-context forward의 signed coefficient 방식은 구현되어 있었다.
8. 반면 기존 몬테카를로 테스트는 작은 손실 스칼라만 검사했고, calibration은 같은 parent 자료로 측정과 선택을 수행해 독립 검증이 없었다.
9. 사용자는 `[1, 0.5, 0.25, 0.125]`를 별도 smoke 설정으로 두고, calibration을 측정 64 / 선택 32 / 감사 32로 구성하기로 결정했다.
10. 사용자는 학습 인자 가독성 개선과 estimator·calibration 보완을 합친 통합 계획의 구현을 요청했다.
11. 통합 계획 구현을 완료했다. 실제 C4와 350M model을 사용한 단일 H100 BF16 64/32/32 calibration이 독립 감사를 통과했고, 최종 tail schedule `[1, 1, 0.75, 0.125]`가 생성됐다.
12. 간헐적인 단일 GPU timing 지연은 parent·level당 3회 측정 중앙값으로 통제하고 원 반복시간을 보존하기로 구현했다.
13. 사용자는 최종 종합 보고서, 최종 estimator 설정, calibration 상세 보고서의 역할과 각 field를 더 쉽고 자세하게 설명해 달라고 요청했다.
14. 사용자는 256차원 CountSketch의 신뢰성, correction 방향성, 512 중단 확률 0, 후보 일정 탐색 방식, SDPA flash 실패 원인을 비판적으로 재검토해 달라고 요청했다.
15. 사용자는 중장기 학습 전 검증에서 CountSketch 없이 full-size gradient를 스트리밍 비교할 수 있는지, parent 64와 Monte Carlo 4,096회의 정확한 의미, 여러 학습 시점의 재측정 비용, level gradient와 correction Gram matrix의 관계, 125개 비단조 후보의 타당성, cuDNN attention 사용 가능성을 질문했다.
16. 사용자는 exact full-gradient 검증에서 parent sample을 크게 늘리고, 논리 batch 128과 단조감소 35개 후보를 사용하기로 결정했다.
17. 사용자는 calibration VRAM 상한을 85%, timing 반복을 1회로 확정했다.
18. 장기 경향성 확인을 위해 같은 초기 weight의 Full과 Russian Roulette을 각각 3,000 update씩 두 H100에서 순차 실행하기로 결정했다.
19. 사용자는 exact-path memory limit을 0.85, timing repeat을 1로 확정하고 이번 작업에서 장기학습까지 진행하도록 요청했다.
20. CountSketch 없는 초기 350M calibration, matched Full/RR 각 3,000 update, 두 arm의 30/300/1500/2700/3000 checkpoint 전 좌표 진단을 모두 완료했다.
21. 고정 일정 `[1,1,0.75,0.25]`는 열 개 checkpoint 독립 감사에서 모두 Q=1 대비 효율 관문을 통과했다. RR은 안정 update 시간이 Full보다 52.65% 짧았지만 같은 3,000 update의 최종 validation loss는 0.05286 높았다.
22. 모든 결과와 해석을 checkpoint 보존 디렉터리의 `FINAL_REPORT.md`에 정리했다.
23. 사용자는 방금 실험의 Markdown·JSON 계열 결과 문서를 저장소의 `results` 디렉터리에도 복사하고, 이후 모든 실험에서 같은 이중 보존 절차를 적용하도록 지시했다.
24. 사용자는 `results`의 파일을 ChatGPT Pro가 읽어 실험을 분석할 예정이므로, 불필요하거나 혼동을 유발하는 파일이 있는지 점검하도록 요청했다.
25. 사용자는 `results` 복사본을 각 실험별 `essential` 필수자료와 `appendix` 상세자료·부록 디렉터리로 나누어 정리하도록 요청했다.

## 현재 상태

확정된 full-gradient 검증 및 3,000-step Full/Russian Roulette 비교 계획의 구현, CPU·GPU·calibration·학습·checkpoint drift gate와 최종 보고를 완료했다. 이번 실험의 원본 결과 문서 75개를 보존한 채 `results/full_gradient_3000_20260721T232425_KST_f57ee32/`를 개요·calibration·공식 Full·공식 RR·공동 분석·보조 실행으로 나누고, 각 주요 실험을 `essential`과 `appendix`로 정리했다.
