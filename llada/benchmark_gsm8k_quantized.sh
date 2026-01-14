#!/bin/bash

# Benchmark GSM8k on base, SmoothQuant, and quantized (W4A8) models.
# Uses prefix caching (use_cache=True) and showspeed=True, and writes
# all console output to a timestamped log file in ./logs.
#
# Usage:
#   bash benchmark_gsm8k_quantized.sh               # run all benchmarks (default)
#   bash benchmark_gsm8k_quantized.sh --only base   # only base model
#   bash benchmark_gsm8k_quantized.sh --only smooth # only SmoothQuant model
#   bash benchmark_gsm8k_quantized.sh --only quant  # only quantized W4A8 model
#   bash benchmark_gsm8k_quantized.sh --only quant_smooth  # only quantized+SmoothQuant model

set -euo pipefail

# Move to the directory containing this script so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Environment needed by eval_llada.py / HF datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
run_base=true
run_smooth=true
run_quant=true
run_quant_smooth=true

usage() {
  echo "Usage:"
  echo "  $0                       # run all benchmarks"
  echo "  $0 --only base           # only base model"
  echo "  $0 --only smooth         # only SmoothQuant model"
  echo "  $0 --only quant          # only quantized W4A8 model"
  echo "  $0 --only quant_smooth   # only quantized+SmoothQuant W4A8 model"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || usage
      # Disable all, then enable only the requested one.
      run_base=false
      run_smooth=false
      run_quant=false
      run_quant_smooth=false
      case "$2" in
        base)
          run_base=true
          ;;
        smooth)
          run_smooth=true
          ;;
        quant)
          run_quant=true
          ;;
        quant_smooth)
          run_quant_smooth=true
          ;;
        *)
          echo "Unknown value for --only: '$2'"
          usage
          ;;
      esac
      shift 2
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

task="gsm8k"
length=256
block_length=32
limit=1000
num_fewshot=5
model_path="GSAI-ML/LLaDA-8B-Instruct"
smoothed_model_path="models/llada_smoothquant.pt"
# Quantized models are saved via `save_pretrained` by quantize_llada_w4a8.py.
# These should point to the output directories used there.
quantized_model_path="models/llada_quantized_w4a8"
quantized_smoothquant_path="models/llada_quantized_w4a8_smoothquant"

# Prepare log file
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/gsm8k_quantized_benchmark_${TIMESTAMP}.log"

{
  echo "============================================================"
  echo "GSM8k Benchmark: Base vs SmoothQuant vs Quantized (W4A8)"
  echo "============================================================"
  echo "Task:          ${task}"
  echo "Model path:    ${model_path}"
  echo "SmoothQuant:   ${smoothed_model_path}"
  echo "Quantized:     ${quantized_model_path}"
  echo "Quantized+SQ:  ${quantized_smoothquant_path}"
  echo "Gen length:    ${length}"
  echo "Block length:  ${block_length}"
  echo "Num few-shot:  ${num_fewshot}"
  echo "Limit:         ${limit}"
  echo "Prefix cache:  use_cache=True"
  echo "Show speed:    show_speed=True"
  echo "Timestamp:     ${TIMESTAMP}"
  echo "============================================================"
  echo ""

  if ${run_base}; then
    echo "=== 1) Base model with prefix cache ==="
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --limit "${limit}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True"
    echo ""
  else
    echo "Skipping base model benchmark (per arguments)."
    echo ""
  fi

  if ${run_smooth}; then
    echo "=== 2) SmoothQuant model with prefix cache ==="
    if [ -f "${smoothed_model_path}" ]; then
      accelerate launch eval_llada.py \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        --limit "${limit}" \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,smoothed_model_path=${smoothed_model_path}"
    else
      echo "  WARNING: SmoothQuant model not found at ${smoothed_model_path}"
      echo "  Skipping SmoothQuant benchmark. Run: python smooth_quant_llada.py"
    fi
    echo ""
  else
    echo "Skipping SmoothQuant benchmark (per arguments)."
    echo ""
  fi

  if ${run_quant}; then
    echo "=== 3) Quantized model (W4A8) with prefix cache ==="
    if [ -d "${quantized_model_path}" ]; then
      echo "  Using quantized model directory at ${quantized_model_path}"
      echo "  (generated by quantize_llada_w4a8.py with save_pretrained/Quanto format)"
      accelerate launch eval_llada.py \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        --limit "${limit}" \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,quantized_model_path=${quantized_model_path}"
    else
      echo "  WARNING: Quantized model not found at ${quantized_model_path}"
      echo "  Skipping quantized benchmark. Run: python quantize_llada_w4a8.py"
    fi
    echo ""
  else
    echo "Skipping quantized benchmark (per arguments)."
    echo ""
  fi

  if ${run_quant_smooth}; then
    echo "=== 4) Quantized + SmoothQuant model (W4A8) with prefix cache ==="
    if [ -d "${quantized_smoothquant_path}" ]; then
      echo "  Using quantized+SmoothQuant model directory at ${quantized_smoothquant_path}"
      echo "  (generated by quantize_llada_w4a8.py --load-smoothed with save_pretrained/Quanto format)"
      accelerate launch eval_llada.py \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,quantized_model_path=${quantized_smoothquant_path}"
    else
      echo "  WARNING: Quantized+SmoothQuant model not found at ${quantized_smoothquant_path}"
      echo "  Skipping quantized+SmoothQuant benchmark. Run: python quantize_llada_w4a8.py --load-smoothed"
    fi
  else
    echo "Skipping quantized+SmoothQuant benchmark (per arguments)."
  fi

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Benchmark complete. Results saved to: ${LOG_FILE}"
