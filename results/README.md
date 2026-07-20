# Production validation results

이 directory는 현재 canonical 구현의 검증 결과만 받는다.

로컬의 임시 검증 산출물은 `/tmp` 아래에 생성한다. 외부 4-GPU 검증은 `scripts/run_external_validation.sh`의 output directory에 생성한다. 검증을 완료한 뒤 필요한 보고서만 이 directory로 복사한다.

이전 구현의 synthetic smoke 산출물은 독립 코드베이스 계약과 섞이지 않도록 다음 경로로 이동했다.

`/data1/keunyoung/for_nfs/workspace/umcg_pretraining_legacy_artifacts_20260719/results`
