#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
readonly OUTPUT_ARGUMENT="${1:?usage: scripts/run_external_validation.sh OUTPUT_DIRECTORY}"
readonly OUTPUT_ROOT="$(python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "${OUTPUT_ARGUMENT}")"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "output directory already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi

if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "4" ]]; then
  echo "exactly four visible GPUs are required" >&2
  exit 2
fi

readonly TOKENIZER_COMMIT="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("t5-base", revision="main").sha)')"
readonly C4_COMMIT="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("allenai/c4", revision="main").sha)')"

mkdir -p "${OUTPUT_ROOT}/matrix" "${OUTPUT_ROOT}/gradient" "${OUTPUT_ROOT}/resume"
cd "${PROJECT_ROOT}"

python -m pytest -q
ruff check src tests ./*.py

run_training_case() {
  local distributed_backend="$1"
  local zero_stage="$2"
  local optimizer="$3"
  local precision="$4"
  local save_directory="$5"
  local backend_arguments=(--distributed_backend "${distributed_backend}")
  if [[ "${distributed_backend}" == "zero" ]]; then
    backend_arguments+=(--zero_stage "${zero_stage}")
  fi
  torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    "${backend_arguments[@]}" \
    --model_backend huggingface \
    --model_config configs/model/llama_tiny_t5_4096.json \
    --tokenizer t5-base \
    --tokenizer_revision "${TOKENIZER_COMMIT}" \
    --precision "${precision}" \
    --attention_backend automatic \
    --estimator_config configs/estimator/russian_roulette_safe_1024.json \
    --gradient_estimator russian_roulette \
    --optimizer "${optimizer}" \
    --scheduler cosine \
    --learning_rate 3e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --batch_size 1 \
    --total_batch_size 4 \
    --num_training_steps 3 \
    --warmup_steps 0 \
    --eval_every 3 \
    --eval_parent_batches 1 \
    --save_every 1 \
    --save_dir "${save_directory}" \
    --c4_source streaming \
    --c4_repo allenai/c4 \
    --c4_revision "${C4_COMMIT}" \
    --workers 1 \
    --seed 777 \
    --wandb_mode disabled \
    --name "validation-${distributed_backend}-${zero_stage}-${optimizer}-${precision}"
}

resume_adamw_bfloat16_case() {
  local distributed_backend="$1"
  local zero_stage="$2"
  local source_directory="$3"
  local save_directory="$4"
  local backend_arguments=(--distributed_backend "${distributed_backend}")
  if [[ "${distributed_backend}" == "zero" ]]; then
    backend_arguments+=(--zero_stage "${zero_stage}")
  fi
  torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    "${backend_arguments[@]}" \
    --model_backend huggingface \
    --model_config configs/model/llama_tiny_t5_4096.json \
    --tokenizer t5-base \
    --tokenizer_revision "${TOKENIZER_COMMIT}" \
    --precision bfloat16 \
    --attention_backend automatic \
    --estimator_config configs/estimator/russian_roulette_safe_1024.json \
    --gradient_estimator russian_roulette \
    --optimizer adamw \
    --scheduler cosine \
    --learning_rate 3e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --batch_size 1 \
    --total_batch_size 4 \
    --num_training_steps 3 \
    --warmup_steps 0 \
    --eval_every 3 \
    --eval_parent_batches 1 \
    --save_every 1 \
    --save_dir "${save_directory}" \
    --c4_source streaming \
    --c4_repo allenai/c4 \
    --c4_revision "${C4_COMMIT}" \
    --workers 1 \
    --seed 777 \
    --wandb_mode disabled \
    --name "resume-${distributed_backend}-${zero_stage}-adamw-bfloat16" \
    --continue_from "${source_directory}/checkpoint-00000002"
}

compare_resume_case() {
  local label="$1"
  local source_directory="$2"
  local resumed_directory="$3"
  local comparison_directory="${OUTPUT_ROOT}/resume/${label}"
  mkdir -p "${comparison_directory}"
  python export_weights_main.py \
    --checkpoint "${source_directory}/checkpoint-00000003" \
    --output "${comparison_directory}/uninterrupted.safetensors"
  python export_weights_main.py \
    --checkpoint "${resumed_directory}/checkpoint-00000003" \
    --output "${comparison_directory}/resumed.safetensors"
  python - \
    "${comparison_directory}/uninterrupted.safetensors" \
    "${comparison_directory}/resumed.safetensors" \
    "${comparison_directory}/comparison.json" <<'PY'
import json
import pathlib
import sys

import torch
from safetensors.torch import load_file

left = load_file(sys.argv[1], device="cpu")
right = load_file(sys.argv[2], device="cpu")
if set(left) != set(right):
    raise AssertionError("resume export keys differ")
maximum_absolute_error = 0.0
for name in left:
    torch.testing.assert_close(left[name], right[name], rtol=1e-6, atol=1e-7)
    maximum_absolute_error = max(
        maximum_absolute_error,
        float((left[name] - right[name]).abs().max()),
    )
pathlib.Path(sys.argv[3]).write_text(
    json.dumps(
        {
            "tensor_count": len(left),
            "maximum_absolute_error": maximum_absolute_error,
            "passed": True,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

for precision in float32 float16 bfloat16; do
  for optimizer in adamw adam sgd sgdm muon adamw_8bit; do
    run_training_case \
      ddp none "${optimizer}" "${precision}" \
      "${OUTPUT_ROOT}/matrix/ddp-${optimizer}-${precision}"
  done
  for optimizer in adamw adam sgd sgdm adamw_8bit; do
    run_training_case \
      fsdp2 none "${optimizer}" "${precision}" \
      "${OUTPUT_ROOT}/matrix/fsdp2-${optimizer}-${precision}"
  done
  for zero_stage in 1 2 3; do
    for optimizer in adamw adam sgd sgdm muon; do
      run_training_case \
        zero "${zero_stage}" "${optimizer}" "${precision}" \
        "${OUTPUT_ROOT}/matrix/zero${zero_stage}-${optimizer}-${precision}"
    done
  done
done

if python - <<'PY'
import importlib.util
import torch

supported = (
    importlib.util.find_spec("torchao") is not None
    and torch.cuda.get_device_capability(0) >= (8, 9)
)
raise SystemExit(0 if supported else 1)
PY
then
  for distributed_backend in ddp fsdp2; do
    for optimizer in adamw adam sgd sgdm; do
      run_training_case \
        "${distributed_backend}" none "${optimizer}" float8 \
        "${OUTPUT_ROOT}/matrix/${distributed_backend}-${optimizer}-float8"
    done
  done
fi

torchrun --standalone --nproc_per_node 1 \
  tests/distributed/global_gradient_worker.py \
  --output "${OUTPUT_ROOT}/gradient/one-process.json"

torchrun --standalone --nproc_per_node 4 \
  tests/distributed/global_gradient_worker.py \
  --output "${OUTPUT_ROOT}/gradient/four-process.json" \
  --reference "${OUTPUT_ROOT}/gradient/one-process.json"

resume_adamw_bfloat16_case \
  ddp none \
  "${OUTPUT_ROOT}/matrix/ddp-adamw-bfloat16" \
  "${OUTPUT_ROOT}/resume/ddp-run"
compare_resume_case \
  ddp \
  "${OUTPUT_ROOT}/matrix/ddp-adamw-bfloat16" \
  "${OUTPUT_ROOT}/resume/ddp-run"

resume_adamw_bfloat16_case \
  fsdp2 none \
  "${OUTPUT_ROOT}/matrix/fsdp2-adamw-bfloat16" \
  "${OUTPUT_ROOT}/resume/fsdp2-run"
compare_resume_case \
  fsdp2 \
  "${OUTPUT_ROOT}/matrix/fsdp2-adamw-bfloat16" \
  "${OUTPUT_ROOT}/resume/fsdp2-run"

for zero_stage in 1 2 3; do
  resume_adamw_bfloat16_case \
    zero "${zero_stage}" \
    "${OUTPUT_ROOT}/matrix/zero${zero_stage}-adamw-bfloat16" \
    "${OUTPUT_ROOT}/resume/zero${zero_stage}-run"
  compare_resume_case \
    "zero${zero_stage}" \
    "${OUTPUT_ROOT}/matrix/zero${zero_stage}-adamw-bfloat16" \
    "${OUTPUT_ROOT}/resume/zero${zero_stage}-run"
done

python - "${OUTPUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
matrix_runs = sorted(path.name for path in (root / "matrix").iterdir() if path.is_dir())
summary = {
    "schema_version": 1,
    "status": "matrix_and_global_gradient_passed",
    "matrix_run_count": len(matrix_runs),
    "matrix_runs": matrix_runs,
    "one_process_gradient": json.loads(
        (root / "gradient/one-process.json").read_text(encoding="utf-8")
    ),
    "four_process_gradient": json.loads(
        (root / "gradient/four-process.json").read_text(encoding="utf-8")
    ),
    "resume_comparisons": {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "resume").glob("*/comparison.json"))
    },
}
(root / "validation_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "matrix, global-gradient, and native-resume validation passed: ${OUTPUT_ROOT}"
