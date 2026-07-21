#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
readonly EXPECTED_CONDA_PREFIX="/home/ubuntu/keunyoung/miniconda3/envs/umcg"
readonly C4_ROOT="/home/ubuntu/data/c4_en/en"
readonly C4_REVISION="1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
readonly TOKENIZER_REVISION="a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
readonly ARTIFACT_PARENT="/home/ubuntu/checkpoint/keunyoung/umcg"
readonly GATES=(cpu single_bf16 distributed_bf16 resume fp8 vram c4_350m)
readonly C4_350M_FULL_RR_ATOL="3e-3"
readonly C4_350M_FULL_RR_RELATIVE_L2_MAX="1e-3"
readonly C4_350M_RESUME_ATOL="5e-4"
readonly C4_350M_VALIDATION_LOSS_ATOL="1e-3"

CAMPAIGN_DIRECTORY=""
FROM_GATE=""
THROUGH_GATE="c4_350m"
DISTRIBUTED_FROM_CASE=""
RESUME_FROM_CASE=""
C4_350M_REUSE_GATE=""

usage() {
  echo "usage: scripts/run_h100_2gpu_validation.sh --campaign-dir DIRECTORY --from-gate GATE [--through-gate GATE] [--distributed-from-case LABEL] [--resume-from-case LABEL] [--c4-350m-reuse-gate DIRECTORY]" >&2
  echo "gates: ${GATES[*]}" >&2
}

while (($#)); do
  case "$1" in
    --campaign-dir)
      CAMPAIGN_DIRECTORY="${2:?missing value for --campaign-dir}"
      shift 2
      ;;
    --from-gate)
      FROM_GATE="${2:?missing value for --from-gate}"
      shift 2
      ;;
    --through-gate)
      THROUGH_GATE="${2:?missing value for --through-gate}"
      shift 2
      ;;
    --distributed-from-case)
      DISTRIBUTED_FROM_CASE="${2:?missing value for --distributed-from-case}"
      shift 2
      ;;
    --resume-from-case)
      RESUME_FROM_CASE="${2:?missing value for --resume-from-case}"
      shift 2
      ;;
    --c4-350m-reuse-gate)
      C4_350M_REUSE_GATE="${2:?missing value for --c4-350m-reuse-gate}"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${CAMPAIGN_DIRECTORY}" || -z "${FROM_GATE}" ]]; then
  usage
  exit 2
fi

readonly CAMPAIGN_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${CAMPAIGN_DIRECTORY}")"
if [[ "${CAMPAIGN_ROOT}" != "${ARTIFACT_PARENT}"/* || ! -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  echo "campaign directory is outside the approved root or lacks campaign.json: ${CAMPAIGN_ROOT}" >&2
  exit 2
fi
if [[ -n "${C4_350M_REUSE_GATE}" ]]; then
  C4_350M_REUSE_GATE="$(
    python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
      "${C4_350M_REUSE_GATE}"
  )"
  case "${C4_350M_REUSE_GATE}" in
    "${CAMPAIGN_ROOT}"/attempt-[0-9][0-9][0-9]/07_c4_350m) ;;
    *)
      echo "C4 reuse gate is not an attempt in the active campaign: ${C4_350M_REUSE_GATE}" >&2
      exit 2
      ;;
  esac
  if [[ ! -d "${C4_350M_REUSE_GATE}" ]]; then
    echo "C4 reuse gate does not exist: ${C4_350M_REUSE_GATE}" >&2
    exit 2
  fi
fi

start_index=-1
end_index=-1
for index in "${!GATES[@]}"; do
  if [[ "${GATES[$index]}" == "${FROM_GATE}" ]]; then
    start_index="${index}"
  fi
  if [[ "${GATES[$index]}" == "${THROUGH_GATE}" ]]; then
    end_index="${index}"
  fi
done
if ((start_index < 0 || end_index < start_index)); then
  usage
  exit 2
fi

actual_prefix="$(python -c 'import sys; print(sys.prefix)')"
if [[ "${actual_prefix}" != "${EXPECTED_CONDA_PREFIX}" ]]; then
  echo "validation must run inside ${EXPECTED_CONDA_PREFIX}; current prefix=${actual_prefix}" >&2
  exit 2
fi

maximum_attempt=0
shopt -s nullglob
for path in "${CAMPAIGN_ROOT}"/attempt-[0-9][0-9][0-9]; do
  name="${path##*/attempt-}"
  value="$((10#${name}))"
  if ((value > maximum_attempt)); then
    maximum_attempt="${value}"
  fi
done
shopt -u nullglob
readonly ATTEMPT_NUMBER="$((maximum_attempt + 1))"
readonly ATTEMPT_DIRECTORY="${CAMPAIGN_ROOT}/attempt-$(printf '%03d' "${ATTEMPT_NUMBER}")"
mkdir "${ATTEMPT_DIRECTORY}"

export HF_HOME="${CAMPAIGN_ROOT}/_cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="${CAMPAIGN_ROOT}/_cache/torchinductor"
export TRITON_CACHE_DIR="${CAMPAIGN_ROOT}/_cache/triton"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

CURRENT_GATE="initialization"
CURRENT_CASE=""
on_error() {
  status=$?
  trap - ERR
  jq -n \
    --arg gate "${CURRENT_GATE}" \
    --argjson exit_code "${status}" \
    --arg from_gate "${FROM_GATE}" \
    --arg through_gate "${THROUGH_GATE}" \
    --arg case_name "${CURRENT_CASE}" \
    '{schema_version:1,status:"failed",gate:$gate,case:(if $case_name == "" then null else $case_name end),exit_code:$exit_code,from_gate:$from_gate,through_gate:$through_gate}' \
    > "${ATTEMPT_DIRECTORY}/failure.json"
  jq '.status = "failed"' "${ATTEMPT_DIRECTORY}/attempt.json" \
    > "${ATTEMPT_DIRECTORY}/attempt.json.tmp"
  mv "${ATTEMPT_DIRECTORY}/attempt.json.tmp" "${ATTEMPT_DIRECTORY}/attempt.json"
  exit "${status}"
}
trap on_error ERR

jq -n \
  --arg from_gate "${FROM_GATE}" \
  --arg through_gate "${THROUGH_GATE}" \
  --arg distributed_from_case "${DISTRIBUTED_FROM_CASE}" \
  --arg resume_from_case "${RESUME_FROM_CASE}" \
  --arg c4_350m_reuse_gate "${C4_350M_REUSE_GATE}" \
  --arg source_commit "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" \
  --arg source_status "$(git -C "${PROJECT_ROOT}" status --short)" \
  '{schema_version:1,status:"running",from_gate:$from_gate,through_gate:$through_gate,distributed_from_case:(if $distributed_from_case == "" then null else $distributed_from_case end),resume_from_case:(if $resume_from_case == "" then null else $resume_from_case end),c4_350m_reuse_gate:(if $c4_350m_reuse_gate == "" then null else $c4_350m_reuse_gate end),source_commit:$source_commit,source_status:$source_status}' \
  > "${ATTEMPT_DIRECTORY}/attempt.json"
git -C "${PROJECT_ROOT}" diff --binary > "${ATTEMPT_DIRECTORY}/source.patch"

should_run() {
  local gate="$1"
  local gate_index=-1
  for index in "${!GATES[@]}"; do
    if [[ "${GATES[$index]}" == "${gate}" ]]; then
      gate_index="${index}"
    fi
  done
  ((gate_index >= start_index && gate_index <= end_index))
}

require_free_gpus() {
  local processes
  processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "${processes}" ]]; then
    echo "GPU compute processes are still active; refusing to continue: ${processes}" >&2
    return 1
  fi
  if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "2" ]]; then
    echo "exactly two visible GPUs are required" >&2
    return 1
  fi
}

backend_arguments() {
  local backend="$1"
  local zero_stage="$2"
  BACKEND_ARGUMENTS=(--distributed_backend "${backend}")
  if [[ "${backend}" == "zero" ]]; then
    BACKEND_ARGUMENTS+=(--zero_stage "${zero_stage}")
  fi
}

assert_training_run() {
  local run_directory="$1"
  local final_step="$2"
  test -f "${run_directory}/resolved_config.json"
  test -f "${run_directory}/run_manifest.json"
  test -s "${run_directory}/metrics.jsonl"
  test -f "${run_directory}/checkpoint-$(printf '%08d' "${final_step}")/COMPLETE"
}

run_tiny_case() {
  local gate_directory="$1"
  local label="$2"
  local backend="$3"
  local zero_stage="$4"
  local optimizer="$5"
  local precision="$6"
  local gradient_estimator="$7"
  local scheduler="$8"
  local steps="$9"
  local run_directory="${gate_directory}/runs/${label}"
  mkdir -p "${gate_directory}/runs" "${gate_directory}/logs"
  backend_arguments "${backend}" "${zero_stage}"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
    "${PROJECT_ROOT}/torchrun_main.py" \
    "${BACKEND_ARGUMENTS[@]}" \
    --model_backend huggingface \
    --model_config "${PROJECT_ROOT}/configs/model/llama_tiny_t5_4096.json" \
    --tokenizer t5-base \
    --tokenizer_revision "${TOKENIZER_REVISION}" \
    --precision "${precision}" \
    --attention_backend automatic \
    --estimator_config "${PROJECT_ROOT}/configs/estimator/russian_roulette_safe_1024.json" \
    --gradient_estimator "${gradient_estimator}" \
    --optimizer "${optimizer}" \
    --scheduler "${scheduler}" \
    --learning_rate 3e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --batch_size 1 \
    --total_batch_size 4 \
    --num_training_steps "${steps}" \
    --warmup_steps 0 \
    --eval_every "${steps}" \
    --eval_parent_batches 1 \
    --save_every 1 \
    --save_dir "${run_directory}" \
    --c4_source local_raw \
    --c4_repo allenai/c4 \
    --c4_revision "${C4_REVISION}" \
    --c4_local_path "${C4_ROOT}" \
    --workers 1 \
    --seed 777 \
    --wandb_mode disabled \
    --name "${label}" \
    > "${gate_directory}/logs/${label}.log" 2>&1
  assert_training_run "${run_directory}" "${steps}"
}

reuse_tiny_case() {
  local gate_directory="$1"
  local label="$2"
  local steps="$3"
  local source_directory
  source_directory="$(
    find "${CAMPAIGN_ROOT}" -mindepth 4 -maxdepth 4 -type d \
      -path "*/03_distributed_bf16/runs/${label}" | sort | tail -n 1
  )"
  if [[ -z "${source_directory}" ]]; then
    echo "no completed prior run is available for distributed case ${label}" >&2
    return 1
  fi
  assert_training_run "${source_directory}" "${steps}"
  mkdir -p "${gate_directory}/runs" "${gate_directory}/reused"
  ln -s "${source_directory}" "${gate_directory}/runs/${label}"
  jq -n --arg case_name "${label}" --arg source "${source_directory}" \
    '{schema_version:1,case_name:$case_name,source:$source}' \
    > "${gate_directory}/reused/${label}.json"
}

run_cpu_gate() {
  local gate_directory="${ATTEMPT_DIRECTORY}/01_cpu"
  mkdir -p "${gate_directory}"
  cd "${PROJECT_ROOT}"
  python -m pytest -q > "${gate_directory}/pytest.log" 2>&1
  ruff check src tests ./*.py > "${gate_directory}/ruff.log" 2>&1
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
    --save_dir "${gate_directory}/smoke" \
    > "${gate_directory}/smoke.log" 2>&1
  jq -e '.finite == true and (.updates | length) == 1' \
    "${gate_directory}/smoke/smoke_report.json" >/dev/null
}

run_single_gate() {
  require_free_gpus
  local gate_directory="${ATTEMPT_DIRECTORY}/02_single_bf16"
  mkdir -p "${gate_directory}/logs"
  "${PROJECT_ROOT}/scripts/run_gpu0_smoke.sh" "${gate_directory}/gpu0-huggingface-full" \
    > "${gate_directory}/logs/gpu0.log" 2>&1
  "${PROJECT_ROOT}/scripts/run_gpu1_smoke.sh" "${gate_directory}/gpu1-reference-rr" \
    > "${gate_directory}/logs/gpu1.log" 2>&1
  jq -e '.finite == true and .precision == "bfloat16" and (.updates | length) == 2' \
    "${gate_directory}/gpu0-huggingface-full/smoke_report.json" >/dev/null
  jq -e '.finite == true and .precision == "bfloat16" and (.updates | length) == 2' \
    "${gate_directory}/gpu1-reference-rr/smoke_report.json" >/dev/null
}

run_distributed_gate() {
  require_free_gpus
  local gate_directory="${ATTEMPT_DIRECTORY}/03_distributed_bf16"
  local started=false
  local matched=false
  local cases=(
    "ddp-adamw-rr|ddp|none|adamw|russian_roulette|cosine"
    "fsdp2-adamw-rr|fsdp2|none|adamw|russian_roulette|cosine"
    "zero1-adamw-rr|zero|1|adamw|russian_roulette|cosine"
    "zero2-adamw-rr|zero|2|adamw|russian_roulette|cosine"
    "zero3-adamw-rr|zero|3|adamw|russian_roulette|cosine"
    "ddp-adamw-full|ddp|none|adamw|full|cosine"
    "ddp-adam-rr|ddp|none|adam|russian_roulette|linear"
    "ddp-sgd-rr|ddp|none|sgd|russian_roulette|cosine"
    "ddp-sgdm-rr|ddp|none|sgdm|russian_roulette|cosine"
    "ddp-muon-rr|ddp|none|muon|russian_roulette|cosine"
    "ddp-adamw_8bit-rr|ddp|none|adamw_8bit|russian_roulette|cosine"
  )
  if [[ -z "${DISTRIBUTED_FROM_CASE}" ]]; then
    started=true
  fi
  for spec in "${cases[@]}"; do
    IFS='|' read -r label backend zero_stage optimizer gradient_estimator scheduler <<< "${spec}"
    if [[ "${label}" == "${DISTRIBUTED_FROM_CASE}" ]]; then
      started=true
      matched=true
    fi
    if [[ "${started}" != true ]]; then
      reuse_tiny_case "${gate_directory}" "${label}" 3
      continue
    fi
    CURRENT_CASE="${label}"
    run_tiny_case \
      "${gate_directory}" "${label}" "${backend}" "${zero_stage}" "${optimizer}" \
      bfloat16 "${gradient_estimator}" "${scheduler}" 3
    CURRENT_CASE=""
  done
  if [[ -n "${DISTRIBUTED_FROM_CASE}" && "${matched}" != true ]]; then
    echo "unknown distributed case: ${DISTRIBUTED_FROM_CASE}" >&2
    return 1
  fi
}

latest_distributed_case_directory() {
  local case_name="$1"
  find "${CAMPAIGN_ROOT}" -mindepth 4 -maxdepth 4 -type d \
    -path "*/03_distributed_bf16/runs/${case_name}" | sort | tail -n 1
}

RESUME_SOURCE_DIRECTORY=""
ensure_resume_source() {
  local gate_directory="$1"
  local case_name="$2"
  local backend="$3"
  local zero_stage="$4"
  local source_directory
  local source_hash=""
  local regenerated=false
  local current_hash
  current_hash="$(
    python -c 'import pathlib,sys; from umcg.config import source_tree_sha256; print(source_tree_sha256(pathlib.Path(sys.argv[1])))' \
      "${PROJECT_ROOT}"
  )"
  source_directory="$(latest_distributed_case_directory "${case_name}")"
  if [[ -n "${source_directory}" ]]; then
    assert_training_run "${source_directory}" 3
    source_hash="$(jq -r '.source_tree_sha256' "${source_directory}/resolved_config.json")"
  fi
  if [[ -z "${source_directory}" || "${source_hash}" != "${current_hash}" ]]; then
    CURRENT_CASE="baseline-${case_name}"
    run_tiny_case \
      "${gate_directory}/baselines" "${case_name}" "${backend}" "${zero_stage}" adamw \
      bfloat16 russian_roulette cosine 3
    CURRENT_CASE=""
    source_directory="${gate_directory}/baselines/runs/${case_name}"
    source_hash="$(jq -r '.source_tree_sha256' "${source_directory}/resolved_config.json")"
    if [[ "${source_hash}" != "${current_hash}" ]]; then
      echo "fresh resume baseline source hash differs from the current tree" >&2
      return 1
    fi
    regenerated=true
  fi
  mkdir -p "${gate_directory}/sources"
  jq -n \
    --arg case_name "${case_name}" \
    --arg source "${source_directory}" \
    --arg source_hash "${source_hash}" \
    --arg current_hash "${current_hash}" \
    --argjson regenerated "${regenerated}" \
    '{schema_version:1,case_name:$case_name,source:$source,source_hash:$source_hash,current_hash:$current_hash,regenerated:$regenerated}' \
    > "${gate_directory}/sources/${case_name}.json"
  RESUME_SOURCE_DIRECTORY="${source_directory}"
}

resume_tiny_case() {
  local gate_directory="$1"
  local source_directory="$2"
  local label="$3"
  local backend="$4"
  local zero_stage="$5"
  local resumed_directory="${gate_directory}/runs/${label}-resumed"
  local comparison_directory="${gate_directory}/comparisons/${label}"
  mkdir -p "${gate_directory}/runs" "${gate_directory}/logs" "${comparison_directory}"
  backend_arguments "${backend}" "${zero_stage}"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
    "${PROJECT_ROOT}/torchrun_main.py" \
    "${BACKEND_ARGUMENTS[@]}" \
    --model_backend huggingface \
    --model_config "${PROJECT_ROOT}/configs/model/llama_tiny_t5_4096.json" \
    --tokenizer t5-base \
    --tokenizer_revision "${TOKENIZER_REVISION}" \
    --precision bfloat16 \
    --attention_backend automatic \
    --estimator_config "${PROJECT_ROOT}/configs/estimator/russian_roulette_safe_1024.json" \
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
    --save_dir "${resumed_directory}" \
    --c4_source local_raw \
    --c4_repo allenai/c4 \
    --c4_revision "${C4_REVISION}" \
    --c4_local_path "${C4_ROOT}" \
    --workers 1 \
    --seed 777 \
    --wandb_mode disabled \
    --name "${label}-resumed" \
    --continue_from "${source_directory}/checkpoint-00000002" \
    > "${gate_directory}/logs/${label}.log" 2>&1
  assert_training_run "${resumed_directory}" 3
  python "${PROJECT_ROOT}/export_weights_main.py" \
    --checkpoint "${source_directory}/checkpoint-00000003" \
    --output "${comparison_directory}/uninterrupted.safetensors" \
    > "${comparison_directory}/export-uninterrupted.log"
  python "${PROJECT_ROOT}/export_weights_main.py" \
    --checkpoint "${resumed_directory}/checkpoint-00000003" \
    --output "${comparison_directory}/resumed.safetensors" \
    > "${comparison_directory}/export-resumed.log"
  python "${PROJECT_ROOT}/scripts/compare_validation_states.py" \
    --mode weights \
    --left "${comparison_directory}/uninterrupted.safetensors" \
    --right "${comparison_directory}/resumed.safetensors" \
    --output "${comparison_directory}/comparison.json" \
    > "${comparison_directory}/comparison.log"
}

reuse_resume_gradient() {
  local gate_directory="$1"
  local source_directory
  source_directory="$(
    find "${CAMPAIGN_ROOT}" -mindepth 3 -maxdepth 3 -type d \
      -path "*/04_resume/gradient" | sort | tail -n 1
  )"
  if [[ -z "${source_directory}" ]]; then
    echo "no completed prior global-gradient result is available" >&2
    return 1
  fi
  test -s "${source_directory}/one-process.json"
  test -s "${source_directory}/two-process.json"
  mkdir -p "${gate_directory}/reused"
  ln -s "${source_directory}" "${gate_directory}/gradient"
  jq -n --arg case_name global-gradient --arg source "${source_directory}" \
    '{schema_version:1,case_name:$case_name,source:$source}' \
    > "${gate_directory}/reused/global-gradient.json"
}

reuse_resume_case() {
  local gate_directory="$1"
  local label="$2"
  local comparison_file=""
  local candidate
  local candidates=()
  mapfile -t candidates < <(
    find "${CAMPAIGN_ROOT}" -type f -path "*/04_resume/comparisons/${label}/comparison.json" \
      | sort -r
  )
  for candidate in "${candidates[@]}"; do
    if jq -e '.passed == true' "${candidate}" >/dev/null; then
      comparison_file="${candidate}"
      break
    fi
  done
  if [[ -z "${comparison_file}" ]]; then
    echo "no completed prior resume comparison is available for ${label}" >&2
    return 1
  fi
  local comparison_directory="${comparison_file%/comparison.json}"
  local source_gate="${comparison_directory%/comparisons/${label}}"
  local resumed_directory="${source_gate}/runs/${label}-resumed"
  assert_training_run "${resumed_directory}" 3
  mkdir -p "${gate_directory}/comparisons" "${gate_directory}/runs" "${gate_directory}/reused"
  ln -s "${comparison_directory}" "${gate_directory}/comparisons/${label}"
  ln -s "${resumed_directory}" "${gate_directory}/runs/${label}-resumed"
  jq -n --arg case_name "${label}" --arg source "${comparison_directory}" \
    '{schema_version:1,case_name:$case_name,source:$source}' \
    > "${gate_directory}/reused/${label}.json"
}

run_resume_gate() {
  require_free_gpus
  local gate_directory="${ATTEMPT_DIRECTORY}/04_resume"
  local started=false
  local matched=false
  local cases=(
    "ddp|ddp-adamw-rr|ddp|none"
    "fsdp2|fsdp2-adamw-rr|fsdp2|none"
    "zero1|zero1-adamw-rr|zero|1"
    "zero2|zero2-adamw-rr|zero|2"
    "zero3|zero3-adamw-rr|zero|3"
  )
  if [[ -z "${RESUME_FROM_CASE}" ]]; then
    started=true
    mkdir -p "${gate_directory}/gradient"
    CURRENT_CASE="global-gradient"
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 \
      "${PROJECT_ROOT}/tests/distributed/global_gradient_worker.py" \
      --output "${gate_directory}/gradient/one-process.json" \
      > "${gate_directory}/gradient/one-process.log" 2>&1
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
      "${PROJECT_ROOT}/tests/distributed/global_gradient_worker.py" \
      --output "${gate_directory}/gradient/two-process.json" \
      --reference "${gate_directory}/gradient/one-process.json" \
      > "${gate_directory}/gradient/two-process.log" 2>&1
    CURRENT_CASE=""
  else
    case "${RESUME_FROM_CASE}" in
      ddp|fsdp2|zero1|zero2|zero3) ;;
      *) echo "unknown resume case: ${RESUME_FROM_CASE}" >&2; return 1 ;;
    esac
    reuse_resume_gradient "${gate_directory}"
  fi
  for spec in "${cases[@]}"; do
    IFS='|' read -r label source_case backend zero_stage <<< "${spec}"
    if [[ "${label}" == "${RESUME_FROM_CASE}" ]]; then
      started=true
      matched=true
    fi
    if [[ "${started}" != true ]]; then
      reuse_resume_case "${gate_directory}" "${label}"
      continue
    fi
    ensure_resume_source "${gate_directory}" "${source_case}" "${backend}" "${zero_stage}"
    CURRENT_CASE="${label}"
    resume_tiny_case \
      "${gate_directory}" "${RESUME_SOURCE_DIRECTORY}" "${label}" "${backend}" \
      "${zero_stage}"
    CURRENT_CASE=""
  done
  if [[ -n "${RESUME_FROM_CASE}" && "${matched}" != true ]]; then
    echo "unknown resume case: ${RESUME_FROM_CASE}" >&2
    return 1
  fi
}

run_fp8_gate() {
  require_free_gpus
  local gate_directory="${ATTEMPT_DIRECTORY}/05_fp8"
  run_tiny_case "${gate_directory}" ddp-adamw-fp8 ddp none adamw float8 russian_roulette cosine 2
  run_tiny_case "${gate_directory}" fsdp2-adamw-fp8 fsdp2 none adamw float8 russian_roulette cosine 2
  jq -e '.fp8.converted_module_count > 0' \
    "${gate_directory}/runs/ddp-adamw-fp8/resolved_config.json" >/dev/null
  jq -e '.fp8.converted_module_count > 0 and .fp8.fsdp_float8_all_gather == true' \
    "${gate_directory}/runs/fsdp2-adamw-fp8/resolved_config.json" >/dev/null
}

run_vram_case() {
  local gate_directory="$1"
  local label="$2"
  local backend="$3"
  local zero_stage="$4"
  local report_directory="${gate_directory}/reports/${label}"
  mkdir -p "${gate_directory}/reports" "${gate_directory}/logs"
  backend_arguments "${backend}" "${zero_stage}"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
    "${PROJECT_ROOT}/vram_check_main.py" \
    "${BACKEND_ARGUMENTS[@]}" \
    --model_backend huggingface \
    --model_config "${PROJECT_ROOT}/configs/model/llama_350m_t5_4096.json" \
    --tokenizer t5-base \
    --tokenizer_revision "${TOKENIZER_REVISION}" \
    --precision bfloat16 \
    --attention_backend automatic \
    --estimator_config "${PROJECT_ROOT}/configs/estimator/russian_roulette_safe_4096.json" \
    --gradient_estimator full \
    --optimizer adamw \
    --scheduler cosine \
    --learning_rate 3e-4 \
    --batch_size auto \
    --total_batch_size 512 \
    --num_training_steps 1 \
    --warmup_steps 0 \
    --eval_every 1 \
    --eval_parent_batches 1 \
    --save_every 1 \
    --save_dir "${report_directory}" \
    --c4_source local_raw \
    --c4_revision "${C4_REVISION}" \
    --c4_local_path "${C4_ROOT}" \
    --workers 1 \
    --seed 777 \
    --activation_checkpointing \
    --wandb_mode disabled \
    --name "vram-${label}" \
    > "${gate_directory}/logs/${label}.log" 2>&1
  jq -e '.batch_selection.batch_size >= 1 and ([.batch_selection.probes[] | select(.passed)] | length) > 0' \
    "${report_directory}/vram_report.json" >/dev/null
}

run_vram_gate() {
  require_free_gpus
  local gate_directory="${ATTEMPT_DIRECTORY}/06_vram"
  run_vram_case "${gate_directory}" ddp ddp none
  run_vram_case "${gate_directory}" fsdp2 fsdp2 none
  for stage in 1 2 3; do
    run_vram_case "${gate_directory}" "zero${stage}" zero "${stage}"
  done
}

C4_350M_BATCH_SIZE=""
run_350m_case() {
  local gate_directory="$1"
  local label="$2"
  local gradient_estimator="$3"
  local continue_from="${4:-}"
  local run_directory="${gate_directory}/runs/${label}"
  local restart_arguments=()
  if [[ -n "${continue_from}" ]]; then
    restart_arguments=(--continue_from "${continue_from}")
  fi
  mkdir -p "${gate_directory}/runs" "${gate_directory}/logs"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
    "${PROJECT_ROOT}/torchrun_main.py" \
    --distributed_backend ddp \
    --model_backend huggingface \
    --model_config "${PROJECT_ROOT}/configs/model/llama_350m_t5_4096.json" \
    --tokenizer t5-base \
    --tokenizer_revision "${TOKENIZER_REVISION}" \
    --precision bfloat16 \
    --attention_backend automatic \
    --estimator_config "${PROJECT_ROOT}/configs/estimator/russian_roulette_safe_4096.json" \
    --gradient_estimator "${gradient_estimator}" \
    --optimizer adamw \
    --scheduler cosine \
    --learning_rate 3e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --batch_size "${C4_350M_BATCH_SIZE}" \
    --total_batch_size 512 \
    --num_training_steps 20 \
    --warmup_steps 2 \
    --eval_every 10 \
    --eval_parent_batches 8 \
    --save_every 10 \
    --save_dir "${run_directory}" \
    --c4_source local_raw \
    --c4_revision "${C4_REVISION}" \
    --c4_local_path "${C4_ROOT}" \
    --workers 1 \
    --seed 777 \
    --use_torch_compile \
    --activation_checkpointing \
    --wandb_mode disabled \
    --name "${label}" \
    "${restart_arguments[@]}" \
    > "${gate_directory}/logs/${label}.log" 2>&1
  assert_training_run "${run_directory}" 20
}

reuse_350m_training() {
  local source_gate="$1"
  local gate_directory="$2"
  local comparison_directory="${gate_directory}/comparisons"
  local current_source_hash
  current_source_hash="$(
    cd "${PROJECT_ROOT}"
    python -c 'from pathlib import Path; from umcg.config import source_tree_sha256; print(source_tree_sha256(Path.cwd()))'
  )"
  mkdir -p "${gate_directory}/runs" "${gate_directory}/logs" \
    "${comparison_directory}" "${gate_directory}/reused"
  test -f "${source_gate}/batch_selection.json"
  ln -s "${source_gate}/batch_selection.json" "${gate_directory}/batch_selection.json"
  for label in full-uninterrupted rr-uninterrupted rr-resumed; do
    local source_run="${source_gate}/runs/${label}"
    local expected_estimator="russian_roulette"
    if [[ "${label}" == "full-uninterrupted" ]]; then
      expected_estimator="full"
    fi
    assert_training_run "${source_run}" 20
    jq -e \
      --arg source_hash "${current_source_hash}" \
      --arg estimator "${expected_estimator}" \
      '.source_tree_sha256 == $source_hash
        and .runtime.gradient_estimator == $estimator
        and .runtime.distributed_backend == "ddp"
        and .runtime.precision == "bfloat16"
        and .runtime.num_training_steps == 20
        and .runtime.total_batch_size == 512
        and .runtime.c4_source == "local_raw"
        and .runtime.seed == 777' \
      "${source_run}/resolved_config.json" >/dev/null
    test -s "${source_gate}/logs/${label}.log"
    test -s "${source_gate}/comparisons/${label}.safetensors"
    test -f "${source_gate}/comparisons/${label}-export.log"
    ln -s "${source_run}" "${gate_directory}/runs/${label}"
    ln -s "${source_gate}/logs/${label}.log" "${gate_directory}/logs/${label}.log"
    ln -s \
      "${source_gate}/comparisons/${label}.safetensors" \
      "${comparison_directory}/${label}.safetensors"
    ln -s \
      "${source_gate}/comparisons/${label}-export.log" \
      "${comparison_directory}/${label}-export.log"
  done
  jq -n \
    --arg source_gate "${source_gate}" \
    --arg source_tree_sha256 "${current_source_hash}" \
    '{schema_version:1,reuse_scope:"completed-training-and-weight-exports",source_gate:$source_gate,source_tree_sha256:$source_tree_sha256}' \
    > "${gate_directory}/reused/c4_350m.json"
}

run_350m_gate() {
  local gate_directory="${ATTEMPT_DIRECTORY}/07_c4_350m"
  local comparison_directory="${gate_directory}/comparisons"
  if [[ -n "${C4_350M_REUSE_GATE}" ]]; then
    reuse_350m_training "${C4_350M_REUSE_GATE}" "${gate_directory}"
    C4_350M_BATCH_SIZE="$(jq -r '.batch_size' "${gate_directory}/batch_selection.json")"
  else
    require_free_gpus
    local vram_report
    vram_report="$(
      find "${CAMPAIGN_ROOT}" -type f -path "*/06_vram/reports/ddp/vram_report.json" \
        | sort | tail -n 1
    )"
    if [[ -z "${vram_report}" ]]; then
      echo "no completed DDP VRAM report is available for the 350M gate" >&2
      return 1
    fi
    jq -e \
      '.distributed_backend == "ddp" and .precision == "bfloat16" and .maximum_context == 4096 and .batch_selection.batch_size >= 1' \
      "${vram_report}" >/dev/null
    C4_350M_BATCH_SIZE="$(jq -r '.batch_selection.batch_size' "${vram_report}")"
    mkdir -p "${comparison_directory}"
    jq -n \
      --arg source "${vram_report}" \
      --argjson batch_size "${C4_350M_BATCH_SIZE}" \
      --argjson accumulation_steps "$((512 / (C4_350M_BATCH_SIZE * 2)))" \
      '{schema_version:1,source_vram_report:$source,batch_size:$batch_size,world_size:2,total_batch_size:512,accumulation_steps:$accumulation_steps}' \
      > "${gate_directory}/batch_selection.json"
    run_350m_case "${gate_directory}" full-uninterrupted full
    run_350m_case "${gate_directory}" rr-uninterrupted russian_roulette
    run_350m_case \
      "${gate_directory}" rr-resumed russian_roulette \
      "${gate_directory}/runs/rr-uninterrupted/checkpoint-00000010"
    for label in full-uninterrupted rr-uninterrupted rr-resumed; do
      python "${PROJECT_ROOT}/export_weights_main.py" \
        --checkpoint "${gate_directory}/runs/${label}/checkpoint-00000020" \
        --output "${comparison_directory}/${label}.safetensors" \
        > "${comparison_directory}/${label}-export.log"
    done
  fi
  if ((512 % (C4_350M_BATCH_SIZE * 2) != 0)); then
    echo "selected DDP batch size does not divide total batch 512" >&2
    return 1
  fi
  CURRENT_CASE="metrics"
  python "${PROJECT_ROOT}/scripts/compare_350m_metrics.py" \
    --full "${gate_directory}/runs/full-uninterrupted/metrics.jsonl" \
    --rr "${gate_directory}/runs/rr-uninterrupted/metrics.jsonl" \
    --resumed "${gate_directory}/runs/rr-resumed/metrics.jsonl" \
    --validation-loss-atol "${C4_350M_VALIDATION_LOSS_ATOL}" \
    --output "${comparison_directory}/metrics.json" \
    > "${comparison_directory}/metrics.log"
  CURRENT_CASE="full-vs-rr"
  python "${PROJECT_ROOT}/scripts/compare_validation_states.py" \
    --mode weights \
    --left "${comparison_directory}/full-uninterrupted.safetensors" \
    --right "${comparison_directory}/rr-uninterrupted.safetensors" \
    --rtol 0 \
    --atol "${C4_350M_FULL_RR_ATOL}" \
    --maximum-relative-l2-error "${C4_350M_FULL_RR_RELATIVE_L2_MAX}" \
    --output "${comparison_directory}/full-vs-rr.json" \
    > "${comparison_directory}/full-vs-rr.log"
  CURRENT_CASE="rr-weights-resume"
  python "${PROJECT_ROOT}/scripts/compare_validation_states.py" \
    --mode weights \
    --left "${comparison_directory}/rr-uninterrupted.safetensors" \
    --right "${comparison_directory}/rr-resumed.safetensors" \
    --rtol 0 \
    --atol "${C4_350M_RESUME_ATOL}" \
    --output "${comparison_directory}/rr-weights-resume.json" \
    > "${comparison_directory}/rr-weights-resume.log"
  CURRENT_CASE="rr-native-resume"
  python "${PROJECT_ROOT}/scripts/compare_validation_states.py" \
    --mode ddp-checkpoints \
    --left "${gate_directory}/runs/rr-uninterrupted/checkpoint-00000020" \
    --right "${gate_directory}/runs/rr-resumed/checkpoint-00000020" \
    --rtol 0 \
    --atol "${C4_350M_RESUME_ATOL}" \
    --output "${comparison_directory}/rr-native-resume.json" \
    > "${comparison_directory}/rr-native-resume.log"
  CURRENT_CASE=""
}

for index in "${!GATES[@]}"; do
  gate="${GATES[$index]}"
  if ! should_run "${gate}"; then
    continue
  fi
  CURRENT_GATE="${gate}"
  case "${gate}" in
    cpu) run_cpu_gate ;;
    single_bf16) run_single_gate ;;
    distributed_bf16) run_distributed_gate ;;
    resume) run_resume_gate ;;
    fp8) run_fp8_gate ;;
    vram) run_vram_gate ;;
    c4_350m) run_350m_gate ;;
  esac
done

trap - ERR
jq -n \
  --arg from_gate "${FROM_GATE}" \
  --arg through_gate "${THROUGH_GATE}" \
  '{schema_version:1,status:"passed",from_gate:$from_gate,through_gate:$through_gate}' \
  > "${ATTEMPT_DIRECTORY}/validation_summary.json"
jq '.status = "passed"' "${ATTEMPT_DIRECTORY}/attempt.json" \
  > "${ATTEMPT_DIRECTORY}/attempt.json.tmp"
mv "${ATTEMPT_DIRECTORY}/attempt.json.tmp" "${ATTEMPT_DIRECTORY}/attempt.json"
echo "validation passed from ${FROM_GATE} through ${THROUGH_GATE}: ${ATTEMPT_DIRECTORY}"
