# UMCG 구현 상태

업데이트: 2026-07-20

## 구현 완료

- 여섯 개 canonical entrypoint
- DDP, PyTorch FSDP2, DeepSpeed ZeRO-1·2·3 경로
- Hugging Face LLaMA와 독립 reference LLaMA
- Single-document C4 chunking과 exact iterator state
- Full과 Russian Roulette global token-average objective
- AdamW, Adam, SGD, SGDM, Muon, TorchAO AdamW 8-bit
- Cosine과 linear scheduler
- FP32, FP16, BF16, TorchAO FP8 policy
- Local-only automatic attention capability test와 benchmark
- DDP, FSDP2, ZeRO native checkpoint
- 공통 FP32 safetensors export
- Calibration, VRAM check, smoke test, local C4 preparation
- Rank 0 W&B와 항상 켜진 JSONL logging

## 현재 로컬 검증

- CPU unit와 integration test 64개: 통과
- Hugging Face/reference logits, loss, gradient 동등성: 통과
- CPU reference smoke update: 통과
- `t5-base` vocabulary 32100, EOS 1, PAD 0, immutable Hub commit 확인: 통과
- TITAN RTX에서 GQA와 오른쪽 padding을 포함한 automatic attention 검사: `eager` 선택
- TITAN RTX 단일 GPU Hugging Face DDP FP16 production 2 update와 full validation: 통과
- TITAN RTX 단일 GPU Hugging Face FSDP2 FP32 production 2 update와 full validation: 통과
- DDP와 FSDP2의 step 1 native resume 후 step 2 replay: 통과
- 두 backend 모두 중단 없는 run과 resume run의 21개 FP32 export 비교: 최대 절대 오차 `0.0`
- 독립 reference LLaMA production update와 Hugging Face export weight 로드: 통과
- Optimizer-free calibration JSON과 report 생성: 통과
- 실제 최대-context VRAM probe: batch divisor `1, 2, 4` 통과, `4` 선택
- LM head chunk backward 재계산 후 해당 tiny run의 peak allocated VRAM: 약 199 MB
- 이전 구현 결과는 `umcg_pretraining_legacy_artifacts_20260719`로 이동

## 아직 완료로 표시하지 않는 항목

- 4-GPU DDP
- 4-GPU FSDP2
- 4-GPU ZeRO-1·2·3
- Multi-GPU optimizer와 precision 전체 matrix
- Hopper 또는 Blackwell FP8
- FlashAttention-2·3·4 실제 package별 선택
- 350M C4 20-update save/resume

위 항목은 현재 장비에서 실행하지 않는다. 외부 서버 검증 절차는 `docs/HANDOFF_TO_EXPERIMENT_SERVER.md`에 있다.
