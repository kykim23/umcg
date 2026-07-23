# Smoke test protocol

Smoke test는 실제 C4 학습과 분리되어 있다. Synthetic parent로 model, estimator, optimizer의 한두 update만 검사한다.

## CPU

```bash
python smoke_main.py \
  --device cpu \
  --model_backend reference \
  --model_config configs/model/llama_tiny_smoke_1024.json \
  --precision float32 \
  --attention_backend eager \
  --estimator_config configs/estimator/russian_roulette_safe_1024.json \
  --gradient_estimator russian_roulette \
  --optimizer adamw \
  --scheduler cosine \
  --learning_rate 1e-3 \
  --batch_size 1 \
  --num_training_steps 1 \
  --warmup_steps 0 \
  --seed 777 \
  --save_dir /tmp/umcg-cpu-smoke
```

CPU는 `float32`만 받는다.

## 단일 GPU

정확히 한 GPU만 보이게 한다.

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_main.py \
  --device cuda \
  --model_backend huggingface \
  --model_config configs/model/llama_tiny_smoke_1024.json \
  --precision bfloat16 \
  --attention_backend automatic \
  --estimator_config configs/estimator/russian_roulette_safe_1024.json \
  --gradient_estimator full \
  --optimizer adamw \
  --scheduler linear \
  --learning_rate 1e-3 \
  --batch_size 1 \
  --num_training_steps 2 \
  --warmup_steps 0 \
  --seed 777 \
  --save_dir /tmp/umcg-gpu-smoke
```

결과는 `smoke_report.json`이다. Scalar와 모든 gradient가 finite여야 한다.

4096 Russian Roulette 분기 자체를 검사할 때에는 `configs/estimator/russian_roulette_smoke_4096.json`과 `configs/model/llama_tiny_t5_4096.json`을 사용한다. 이 estimator는 `[1, 0.5, 0.25, 0.125]`를 사용하지만 `source.type=smoke_unvalidated`, `efficiency_claim=false`이며 최종 학습 설정으로 승격하지 않는다. 최종 일정은 `calibrate_main.py`의 측정 64 / 선택 32 / 독립 감사 32 결과로만 만든다.

H100 검증에서는 FP16을 실행하지 않고 BF16을 사용한다. FP8은 본 smoke와 분리된 2-GPU 기능 관문에서 검증한다.

## CPU test suite

```bash
pytest -q
ruff check src tests *.py
```

이 검사는 production C4, multi-GPU collective, DeepSpeed shard를 대신하지 않는다.

## VRAM 검사

VRAM 검사는 실제 distributed backend와 최대 context를 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 \
torchrun --standalone --nproc_per_node 4 vram_check_main.py \
  --distributed_backend fsdp2 \
  --model_backend huggingface \
  --model_config configs/model/llama_350m_t5_4096.json \
  --precision bfloat16 \
  --attention_backend automatic \
  --estimator_config configs/estimator/russian_roulette_safe_4096.json \
  --gradient_estimator full \
  --optimizer adamw \
  --scheduler cosine \
  --learning_rate 3e-4 \
  --batch_size auto \
  --total_batch_size 512 \
  --num_training_steps 1 \
  --warmup_steps 0 \
  --eval_every 1 \
  --save_every 1 \
  --save_dir /tmp/umcg-vram-check \
  --c4_source streaming \
  --seed 777 \
  --wandb_mode disabled \
  --name umcg-vram-check
```
