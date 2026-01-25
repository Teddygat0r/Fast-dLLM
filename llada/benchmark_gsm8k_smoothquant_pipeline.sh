#!/bin/bash

# Benchmark GSM8k using the SmoothQuant pipeline (smoothquant folder with QuantLinear layers).
# This applies smoothing + W8A8 quantization via the apply_smoothquant_pipeline function.
#
# Uses prefix caching (use_cache=True) and show_speed=True, and writes
# all console output to a timestamped log file in ./logs.
#
# Usage:
#   bash benchmark_gsm8k_smoothquant_pipeline.sh                    # run all benchmarks (default)
#   bash benchmark_gsm8k_smoothquant_pipeline.sh --only base        # only base model (no quantization)
#   bash benchmark_gsm8k_smoothquant_pipeline.sh --only smoothquant # only SmoothQuant pipeline W8A8
#   bash benchmark_gsm8k_smoothquant_pipeline.sh --only smooth_only # only smoothing (no quantization)
#   bash benchmark_gsm8k_smoothquant_pipeline.sh --limit 10         # limit number of GSM8k questions
#   bash benchmark_gsm8k_smoothquant_pipeline.sh --batch-size 4     # set batch size
#   bash benchmark_gsm8k_smoothquant_pipeline.sh --alpha 0.6        # set SmoothQuant alpha

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
run_smoothquant=true
run_smooth_only=true

task="gsm8k"
length=256
block_length=32
limit=""  # Empty means no limit (full dataset)
num_fewshot=5
batch_size=1
model_path="GSAI-ML/LLaDA-8B-Instruct"

# Prefix cache setting
use_prefix_cache=true

# SmoothQuant pipeline parameters
smoothquant_alpha=0.5
smoothquant_w_bits=8
smoothquant_a_bits=8
smoothquant_calibration_samples=128
smoothquant_scales_path="models/act_scales_pipeline.pt"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  echo "Benchmark GSM8k with SmoothQuant Pipeline (QuantLinear layers)"
  echo ""
  echo "Usage:"
  echo "  $0                           # run all benchmarks"
  echo "  $0 --only base               # only base model (no quantization)"
  echo "  $0 --only smoothquant        # only SmoothQuant pipeline W8A8"
  echo "  $0 --only smooth_only        # only smoothing (skip quantization)"
  echo ""
  echo "Options:"
  echo "  --limit N                    # limit to N GSM8k questions (default: full dataset)"
  echo "  --batch-size N               # batch size for evaluation (default: 1)"
  echo "  --alpha FLOAT                # SmoothQuant alpha (default: 0.5)"
  echo "  --calibration-samples N      # calibration samples (default: 64)"
  echo "  --scales-path PATH           # path to save/load activation scales"
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
      run_smoothquant=false
      run_smooth_only=false
      case "$2" in
        base)
          run_base=true
          ;;
        smoothquant)
          run_smoothquant=true
          ;;
        smooth_only)
          run_smooth_only=true
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
    --alpha)
      [[ $# -ge 2 ]] || usage
      smoothquant_alpha="$2"
      shift 2
      ;;
    --calibration-samples)
      [[ $# -ge 2 ]] || usage
      smoothquant_calibration_samples="$2"
      shift 2
      ;;
    --scales-path)
      [[ $# -ge 2 ]] || usage
      smoothquant_scales_path="$2"
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
LOG_FILE="${LOG_DIR}/gsm8k_smoothquant_pipeline_${TIMESTAMP}.log"

{
  echo "============================================================"
  echo "GSM8k Benchmark: SmoothQuant Pipeline (QuantLinear layers)"
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
  echo "SmoothQuant Pipeline Config:"
  echo "  Alpha:           ${smoothquant_alpha}"
  echo "  W bits:          ${smoothquant_w_bits}"
  echo "  A bits:          ${smoothquant_a_bits}"
  echo "  Calib samples:   ${smoothquant_calibration_samples}"
  echo "  Scales path:     ${smoothquant_scales_path}"
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

  if ${run_smoothquant}; then
    echo "=== 2) SmoothQuant Pipeline W${smoothquant_w_bits}A${smoothquant_a_bits} with prefix cache ==="
    echo "  Using smoothquant folder's apply_smoothquant_pipeline with QuantLinear layers"
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      ${limit_arg} \
      --batch_size "${batch_size}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=${use_prefix_cache^},show_speed=True,use_smoothquant_pipeline=True,smoothquant_alpha=${smoothquant_alpha},smoothquant_w_bits=${smoothquant_w_bits},smoothquant_a_bits=${smoothquant_a_bits},smoothquant_calibration_samples=${smoothquant_calibration_samples},smoothquant_scales_path=${smoothquant_scales_path}"
    echo ""
  else
    echo "Skipping SmoothQuant pipeline benchmark (per arguments)."
    echo ""
  fi

  if ${run_smooth_only}; then
    echo "=== 3) Smoothing Only (no quantization) with prefix cache ==="
    echo "  Using smoothquant folder's apply_smoothquant_pipeline with skip_quantization=True"
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      ${limit_arg} \
      --batch_size "${batch_size}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=${use_prefix_cache^},show_speed=True,use_smoothquant_pipeline=True,smoothquant_alpha=${smoothquant_alpha},smoothquant_calibration_samples=${smoothquant_calibration_samples},smoothquant_scales_path=${smoothquant_scales_path},smoothquant_skip_quantization=True"
    echo ""
  else
    echo "Skipping smooth-only benchmark (per arguments)."
    echo ""
  fi

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Benchmark complete. Results saved to: ${LOG_FILE}"
