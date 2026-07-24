# 초기 Calibration

Calibration은 학습 시작 전에 각 문맥 길이의 그래디언트와 비용을 측정하고 Russian Roulette tail schedule을 선택·감사하는 과정이다.

- `essential/`: 350M 모델, 논리 parent batch 128, 측정/선택/감사 64/32/32의 공식 full-coordinate 결과와 최종 estimator 설정
- `appendix/smoke_and_pilot/`: 기능과 메모리 경로를 확인한 작은 사전 실행

최종 schedule 판단에는 `essential`만 사용한다.
