#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
readonly OUTPUT_ARGUMENT="${1:?usage: scripts/run_gpu0_smoke.sh OUTPUT_DIRECTORY}"
readonly OUTPUT_ROOT="$(python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "${OUTPUT_ARGUMENT}")"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "output directory already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=0 python smoke_main.py \
  --device cuda \
  --model_backend huggingface \
  --model_config configs/model/llama_tiny_smoke_1024.json \
  --precision bfloat16 \
  --attention_backend automatic \
  --estimator_config configs/estimator/russian_roulette_safe_1024.json \
  --gradient_estimator full \
  --optimizer adamw \
  --scheduler cosine \
  --learning_rate 1e-3 \
  --batch_size 1 \
  --num_training_steps 2 \
  --warmup_steps 0 \
  --seed 777 \
  --save_dir "${OUTPUT_ROOT}"
