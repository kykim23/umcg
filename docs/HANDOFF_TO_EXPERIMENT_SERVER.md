# 외부 4-GPU 검증 handoff

현재 저장소에서는 multi-GPU 결과를 통과로 표시하지 않는다. 아래 절차는 4개의 같은 GPU가 보이는 외부 서버에서 실행한다.

## 1. 환경

```bash
pip install -e '.[dev,distributed,low-precision]'
scripts/inspect_umcg_env.sh
```

DeepSpeed 0.19.3 이상과 TorchAO 0.17 이상을 사용한다. FlashAttention package는 검증할 GPU에 맞는 버전을 설치한다.

Triton용 C/C++ compiler와 binutils가 현재 shell에서 보여야 한다. `scripts/inspect_umcg_env.sh`의 `build_tools`에 누락된 값이 있으면 먼저 환경을 고친다.

## 2. 자동 matrix

정확히 네 GPU만 보이게 한 뒤 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 \
scripts/run_external_validation.sh /absolute/output/umcg-validation
```

이 스크립트는 다음을 실행한다.

- DDP의 허용 optimizer를 FP32, FP16, BF16으로 3 update
- FSDP2의 허용 optimizer를 FP32, FP16, BF16으로 3 update
- ZeRO-1·2·3의 허용 optimizer를 FP32, FP16, BF16으로 3 update
- FP8 가능 GPU에서는 DDP와 FSDP2의 허용 optimizer를 FP8으로 3 update
- 1-process accumulation과 4-process padded global batch gradient 비교
- 매 run의 evaluation과 native checkpoint
- DDP, FSDP2, ZeRO-1·2·3의 step 2 resume와 step 3 weight 비교

한 run이라도 실패하면 스크립트는 종료한다. 성공하면 `validation_summary.json`이 생긴다.

## 3. Native save/resume

자동 matrix가 각 backend의 `checkpoint-00000002`에서 새 directory로 이어서 step 3까지 실행한다. 처음 run과 resume run의 마지막 weight를 각각 export한다.

```bash
python export_weights_main.py \
  --checkpoint /output/matrix/fsdp2-adamw-bfloat16/checkpoint-00000003 \
  --output /output/resume/fsdp2-uninterrupted.safetensors
```

내부 resume 명령은 원래 명령과 같은 scientific 설정을 쓴다. `--save_dir`, `--name`만 새 값으로 바꾸고 checkpoint 경로를 지정한다.

```bash
--continue_from /output/matrix/fsdp2-adamw-bfloat16/checkpoint-00000002
```

DDP, FSDP2, ZeRO-1, ZeRO-2, ZeRO-3에서 각각 비교한다. 결과는 `resume/*/comparison.json`에 남는다. 모든 FP32 tensor가 허용 오차 안에서 같아야 한다.

## 4. Backend와 model backend 전환

DDP, FSDP2, ZeRO-3 checkpoint를 각각 export한다. Export 결과를 다른 distributed backend와 다른 model backend의 `--initial_weights`로 넣는다.

새 run은 optimizer update 0에서 시작해야 한다. Logits와 첫 full validation loss를 비교한다.

## 5. Attention

설치한 FlashAttention package마다 명시 실행을 한 번 한다. 명시 실행은 fallback이 없어야 한다.

- Ampere 또는 Ada: FlashAttention-2
- Hopper: FlashAttention-2, 3, 4 중 실제 설치한 package
- Blackwell: FlashAttention-4와 설치한 이전 package
- Turing: `automatic`이 PyTorch SDPA 또는 eager를 골라야 함

각 run의 `resolved_config.json`에서 다음을 확인한다.

- 모든 rank의 probe 성공 여부
- level별 시간
- 가장 느린 rank 기준 선택
- package version
- right-padding correctness 결과

## 6. VRAM

`vram_check_main.py`를 1024와 4096 estimator preset으로 실행한다. DDP, FSDP2, ZeRO-1·2·3을 각각 검사한다.

선택된 batch는 전체 batch의 divisor여야 한다. 모든 rank의 peak allocated memory가 device total의 90% 이하여야 한다.

## 7. 350M C4 20 update

`configs/model/llama_350m_t5_4096.json`과 4096 preset을 사용한다. Full과 RR에 같은 seed, C4 commit, total batch, optimizer, scheduler, precision, attention을 준다.

먼저 20 update를 중단 없이 실행한다. `save_every=10`, `eval_every=10`을 쓴다. 그다음 step 10 checkpoint에서 새 directory로 resume해 step 20까지 실행한다.

두 step 20 checkpoint를 export한다. Weight, optimizer update, scheduler, C4 iterator, RR sampler, validation loss가 같아야 한다.

## 8. 승인

다음 자료가 모두 있어야 한 조합을 통과로 표시한다.

- 전체 canonical 명령
- package와 GPU 정보
- `resolved_config.json`
- `run_manifest.json`
- `metrics.jsonl`
- Complete native checkpoint
- Export 결과와 비교 보고서

검증하지 않은 조합은 지원 완료 표에 넣지 않는다.
