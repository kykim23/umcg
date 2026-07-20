#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=1 python smoke_main.py \
  --device cuda \
  --model_backend reference \
  --model_config configs/model/llama_tiny_smoke_1024.json \
  --precision float16 \
  --attention_backend automatic \
  --estimator_config configs/estimator/russian_roulette_safe_1024.json \
  --gradient_estimator russian_roulette \
  --optimizer adamw \
  --scheduler linear \
  --learning_rate 1e-3 \
  --batch_size 1 \
  --num_training_steps 2 \
  --warmup_steps 0 \
  --seed 777 \
  --save_dir /tmp/umcg-gpu1-smoke
