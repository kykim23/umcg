# 공식 Russian Roulette 3,000-update 실험

이 디렉터리는 사전에 고정한 tail schedule을 사용한 공식 Russian Roulette 실험이다.

- `essential/metrics.jsonl`: 3,000개 update의 원시 학습·평가 지표와 선택 문맥 통계
- `essential/resolved_config.json`: 실제 적용된 설정
- `essential/run_manifest.json`: 실행 환경과 산출물 정보
- `appendix/checkpoints/`: 30, 300, 1,500, 2,700, 3,000 update checkpoint manifest

직접 비교 대상은 `02_full_3000_matched`이다.
