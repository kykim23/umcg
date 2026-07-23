# 보조·사전 점검·실패 실행

이 디렉터리의 자료는 공식 Full 대 Russian Roulette 결론을 계산할 때 기본 입력으로 사용하지 않는다.

- `01_full_3000_original`: DDP graph-splitting 조건이 공식 RR 실행과 다르므로 보조 기준선
- `02_full_4_update_benchmark`: 실행 조건을 맞추기 위한 짧은 속도 점검
- `03_full_preflight`~`07_rr_no_ddp_split_preflight`: 기능·shape·compile 경로 사전 점검
- `08_rr_failed_aot_literal`, `09_rr_failed_inductor_symbol`: 실패한 compile 접근의 설정과 manifest
- `99_implementation_notes`: 실패 원인을 조사하며 작성한 구현 메모

각 실행에서 `essential`은 그 실행 자체를 이해하는 최소 자료이고, `appendix`는 checkpoint manifest 같은 상세자료다. 여기서 말하는 `essential`은 전체 최종 비교의 필수자료라는 뜻이 아니다.
