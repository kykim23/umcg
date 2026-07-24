# H100 2-GPU 검증 절차

이 절차는 `/home/ubuntu/keunyoung/miniconda3/envs/umcg`와 NVIDIA H100 80GB 두 개를 사용한다. 새 Conda 환경을 만들지 않는다.

## 고정 입력

- C4 원본: `/home/ubuntu/data/c4_en/en`
- C4 upstream commit: `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`
- `t5-base` commit: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`
- 원본자료 root: `/home/ubuntu/checkpoint/keunyoung/umcg`
- GPU 본 검증 precision: BF16
- FP8: 별도 차단 관문
- H100 FP16: 실행하지 않음

`local_raw`는 train `json.gz` 1,024개와 validation `json.gz` 8개를 직접 streaming한다. 원본을 복제하거나 tokenized parent 전체를 만들지 않는다.

## 실행기

반드시 `umcg` 환경에서 실행한다.

```bash
conda activate umcg
scripts/run_h100_2gpu_validation.sh \
  --campaign-dir /home/ubuntu/checkpoint/keunyoung/umcg/validation_TIMESTAMP_COMMIT \
  --from-gate cpu \
  --through-gate cpu
```

허용 gate 순서는 다음과 같다.

1. `cpu`
2. `single_bf16`
3. `distributed_bf16`
4. `resume`
5. `fp8`
6. `vram`
7. `c4_350m`

실패하면 새 attempt에서 실패 gate를 지정한다. 공통 코드나 환경을 바꾸지 않은 한 앞선 gate는 반복하지 않는다.

```bash
scripts/run_h100_2gpu_validation.sh \
  --campaign-dir /absolute/campaign \
  --from-gate resume \
  --through-gate c4_350m
```

350M 학습이 모두 끝난 뒤 비교 단계만 실패했다면 완료된 학습과 weight export를 새 attempt에서 명시적으로 재사용한다. 원래 실패 attempt는 변경하지 않는다.

```bash
scripts/run_h100_2gpu_validation.sh \
  --campaign-dir /absolute/campaign \
  --from-gate c4_350m \
  --through-gate c4_350m \
  --c4-350m-reuse-gate /absolute/campaign/attempt-NNN/07_c4_350m
```

실행기는 GPU 프로세스를 종료하지 않는다. GPU gate 전에 운영자가 GPU 점유를 확인하고, 승인된 프로세스만 별도로 정리해야 한다. 두 GPU에 compute process가 남아 있으면 실행기는 중단한다.

## 합격 자료

각 학습 run에는 다음이 있어야 한다.

- canonical CLI가 포함된 `resolved_config.json`
- `run_manifest.json`
- `metrics.jsonl`
- `COMPLETE` marker가 있는 native checkpoint
- stdout과 stderr를 합친 원본 log
- resume 및 full 대 `Q=1` Russian Roulette 비교 JSON

FP8 gate는 DDP와 FSDP2 모두에서 변환된 module과 TorchAO metadata를 확인한다. 350M gate는 앞선 DDP VRAM gate가 선택한 micro-batch를 고정하고, 전체 batch 512, 20 update, step 10 저장·평가와 step 20 비교를 사용한다.

`Q=1`에서 full과 Russian Roulette 목적함수는 수학적으로 같다. 다만 H100의 BF16 cuDNN attention을 서로 다른 프로세스에서 실행하면 같은 계산 경로끼리도 bitwise deterministic하지 않다. 따라서 350M gate는 다음을 함께 요구한다.

- update 번호, C4 target 수, 누적 token 수, context 선택, learning-rate schedule, precision, backend와 source hash는 정확히 일치
- step 10·20 full validation loss의 절대 오차는 `1e-3` 이하
- full 대 `Q=1` RR exported FP32 weight는 상대 L2 오차 `1e-3` 이하이면서 모든 원소의 절대 오차 `3e-3` 이하
- RR 중단 없는 실행 대 재개 실행의 exported weight와 native checkpoint 부동소수점 tensor는 상대 허용치 0, 절대 허용치 `5e-4` 이하
- checkpoint의 정수 tensor, C4 iterator, sampler count, RNG byte state와 다른 비부동소수점 상태는 정확히 일치

작은 deterministic resume 관문은 별도로 엄격한 `1e-6` 상대·`1e-7` 절대 허용치를 유지한다. 350M 허용치는 그 검사를 대체하지 않는다.
