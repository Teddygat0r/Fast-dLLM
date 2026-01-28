#!/bin/bash

# Benchmark GSM8k using the DuQuant pipeline.
# This applies DuQuant quantization (rotation + permutation + quantization) via the apply_duquant_pipeline function.
#
# Uses prefix caching (use_cache=True) and show_speed=True, and writes
# all console output to a timestamped log file in ./logs.
#
# Usage:
#   bash benchmark_gsm8k_duquant_pipeline.sh                    # run all benchmarks (default)
#   bash benchmark_gsm8k_duquant_pipeline.sh --only base        # only base model (no quantization)
#   bash benchmark_gsm8k_duquant_pipeline.sh --only duquant     # only DuQuant pipeline W8A8
#   bash benchmark_gsm8k_duquant_pipeline.sh --limit 10          # limit number of GSM8k questions
#   bash benchmark_gsm8k_duquant_pipeline.sh --batch-size 4      # set batch size
#   bash benchmark_gsm8k_duquant_pipeline.sh --n-bits 4           # set weight quantization bits
#   bash benchmark_gsm8k_duquant_pipeline.sh --block-size 64     # set DuQuant block size

set -euo pipefail

# Move to the directory containing this script so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Environment needed by eval_llada.py / HF datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
run_base=true
run_duquant=true

task="gsm8k"
length=256
block_length=32
limit=""  # Empty means no limit (full dataset)
num_fewshot=5
batch_size=1
model_path="GSAI-ML/LLaDA-8B-Instruct"

# Prefix cache setting
use_prefix_cache=true

# DuQuant pipeline parameters
duquant_n_bits=8
duquant_a_bits=8
duquant_block_size=128
duquant_max_rotation_step=256
duquant_permutation_times=1
duquant_calibration_samples=128
duquant_seq_len=512
duquant_batch_size=1

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  echo "Benchmark GSM8k with DuQuant Pipeline"
  echo ""
  echo "Usage:"
  echo "  $0                           # run all benchmarks"
  echo "  $0 --only base               # only base model (no quantization)"
  echo "  $0 --only duquant             # only DuQuant pipeline W8A8"
  echo ""
  echo "Options:"
  echo "  --limit N                    # limit to N GSM8k questions (default: full dataset)"
  echo "  --batch-size N               # batch size for evaluation (default: 1)"
  echo "  --n-bits N                   # weight quantization bits (default: 8)"
  echo "  --a-bits N                   # activation quantization bits (default: 8)"
  echo "  --block-size N               # DuQuant block size (default: 128)"
  echo "  --max-rotation-step N         # max rotation iterations (default: 256)"
  echo "  --permutation-times N        # permutation iterations (default: 1)"
  echo "  --calibration-samples N      # calibration samples (default: 128)"
  echo "  --seq-len N                  # sequence length for calibration (default: 512)"
  echo "  --calib-batch-size N         # batch size for calibration (default: 1)"
  echo "  --no-prefix-cache            # disable prefix cache (use_cache=False)"
  echo "  -h, --help                   # show this help"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || usage
      # Disable all, then enable only the requested one.
      run_base=false
      run_duquant=false
      case "$2" in
        base)
          run_base=true
          ;;
        duquant)
          run_duquant=true
          ;;
        *)
          echo "Unknown value for --only: '$2'"
          usage
          ;;
      esac
      shift 2
      ;;
    --limit)
      [[ $# -ge 2 ]] || usage
      limit="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || usage
      batch_size="$2"
      shift 2
      ;;
    --n-bits)
      [[ $# -ge 2 ]] || usage
      duquant_n_bits="$2"
      shift 2
      ;;
    --a-bits)
      [[ $# -ge 2 ]] || usage
      duquant_a_bits="$2"
      shift 2
      ;;
    --block-size)
      [[ $# -ge 2 ]] || usage
      duquant_block_size="$2"
      shift 2
      ;;
    --max-rotation-step)
      [[ $# -ge 2 ]] || usage
      duquant_max_rotation_step="$2"
      shift 2
      ;;
    --permutation-times)
      [[ $# -ge 2 ]] || usage
      duquant_permutation_times="$2"
      shift 2
      ;;
    --calibration-samples)
      [[ $# -ge 2 ]] || usage
      duquant_calibration_samples="$2"
      shift 2
      ;;
    --seq-len)
      [[ $# -ge 2 ]] || usage
      duquant_seq_len="$2"
      shift 2
      ;;
    --calib-batch-size)
      [[ $# -ge 2 ]] || usage
      duquant_batch_size="$2"
      shift 2
      ;;
    --no-prefix-cache)
      use_prefix_cache=false
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: '$1'"
      usage
      ;;
  esac
done

# Build limit argument if specified
limit_arg=""
if [[ -n "${limit}" ]]; then
  limit_arg="--limit ${limit}"
fi

# Prepare log file
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/gsm8k_duquant_pipeline_${TIMESTAMP}.log"

{
  echo "============================================================"
  echo "GSM8k Benchmark: DuQuant Pipeline"
  echo "============================================================"
  echo "Task:              ${task}"
  echo "Model path:        ${model_path}"
  echo "Gen length:        ${length}"
  echo "Block length:      ${block_length}"
  echo "Batch size:        ${batch_size}"
  echo "Num few-shot:      ${num_fewshot}"
  echo "Limit:             ${limit:-'full dataset'}"
  echo "Prefix cache:      use_cache=${use_prefix_cache^}"
  echo "Show speed:        show_speed=True"
  echo ""
  echo "DuQuant Pipeline Config:"
  echo "  Weight bits:     ${duquant_n_bits}"
  echo "  Activation bits: ${duquant_a_bits}"
  echo "  Block size:      ${duquant_block_size}"
  echo "  Max rot steps:   ${duquant_max_rotation_step}"
  echo "  Permutation:     ${duquant_permutation_times}"
  echo "  Calib samples:   ${duquant_calibration_samples}"
  echo "  Seq len:         ${duquant_seq_len}"
  echo "  Calib batch:     ${duquant_batch_size}"
  echo ""
  echo "Timestamp:         ${TIMESTAMP}"
  echo "============================================================"
  echo ""

  if ${run_base}; then
    echo "=== 1) Base model (no quantization) with prefix cache ==="
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      ${limit_arg} \
      --batch_size "${batch_size}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=${use_prefix_cache^},show_speed=True"
    echo ""
  else
    echo "Skipping base model benchmark (per arguments)."
    echo ""
  fi

  if ${run_duquant}; then
    echo "=== 2) DuQuant Pipeline W${duquant_n_bits}A${duquant_a_bits} with prefix cache ==="
    echo "  Using duquant folder's apply_duquant_pipeline"
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      ${limit_arg} \
      --batch_size "${batch_size}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=${use_prefix_cache^},show_speed=True,use_duquant_pipeline=True,duquant_n_bits=${duquant_n_bits},duquant_a_bits=${duquant_a_bits},duquant_block_size=${duquant_block_size},duquant_max_rotation_step=${duquant_max_rotation_step},duquant_permutation_times=${duquant_permutation_times},duquant_calibration_samples=${duquant_calibration_samples},duquant_seq_len=${duquant_seq_len},duquant_batch_size=${duquant_batch_size}"
    echo ""
  else
    echo "Skipping DuQuant pipeline benchmark (per arguments)."
    echo ""
  fi

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Benchmark complete. Results saved to: ${LOG_FILE}"
