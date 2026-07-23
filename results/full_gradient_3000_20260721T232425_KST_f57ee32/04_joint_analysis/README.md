# Full과 Russian Roulette 공동 분석

- `essential/`: 장기학습 비교, checkpoint 통합 진단, schedule drift, 가중치 이동, CSV 표와 SVG 그래프
- `appendix/checkpoint_diagnostics/`: Full과 Russian Roulette 각각 다섯 시점의 full-coordinate 개별 진단 보고서

일반 분석은 `essential`만으로 충분하다. 특정 checkpoint의 Gram matrix, cosine similarity, 독립 감사 통계를 재검토할 때만 `appendix`를 읽는다.
