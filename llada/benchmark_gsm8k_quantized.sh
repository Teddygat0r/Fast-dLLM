#!/bin/bash

# Benchmark GSM8k on base, SmoothQuant, and quantized (W4A8) models.
# Uses prefix caching (use_cache=True) and showspeed=True, and writes
# all console output to a timestamped log file in ./logs.

set -euo pipefail

# Move to the directory containing this script so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Environment needed by eval_llada.py / HF datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task="gsm8k"
length=256
block_length=32
num_fewshot=5
model_path="GSAI-ML/LLaDA-8B-Instruct"
smoothed_model_path="models/llada_smoothquant.pt"
quantized_model_path="models/llada_quantized_w4a8_full.pt"
quantized_smoothquant_path="models/llada_quantized_w4a8_smoothquant_full.pt"

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
  echo "Prefix cache:  use_cache=True"
  echo "Show speed:    show_speed=True"
  echo "Timestamp:     ${TIMESTAMP}"
  echo "============================================================"
  echo ""

  echo "=== 1) Base model with prefix cache ==="
  accelerate launch eval_llada.py \
    --tasks "${task}" \
    --num_fewshot "${num_fewshot}" \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True"

  echo ""
  echo "=== 2) SmoothQuant model with prefix cache ==="
  if [ -f "${smoothed_model_path}" ]; then
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,smoothed_model_path=${smoothed_model_path}"
  else
    echo "  WARNING: SmoothQuant model not found at ${smoothed_model_path}"
    echo "  Skipping SmoothQuant benchmark. Run: python smooth_quant_llada.py"
  fi

  echo ""
  echo "=== 3) Quantized model (W4A8) with prefix cache ==="
  if [ -f "${quantized_model_path}" ]; then
    echo "  Note: Quantized models need to be loaded differently in eval_llada.py"
    echo "  This benchmark assumes eval_llada.py supports quantized_model_path parameter"
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,quantized_model_path=${quantized_model_path}"
  else
    echo "  WARNING: Quantized model not found at ${quantized_model_path}"
    echo "  Skipping quantized benchmark. Run: python quantize_llada_w4a8.py"
  fi

  echo ""
  echo "=== 4) Quantized + SmoothQuant model (W4A8) with prefix cache ==="
  if [ -f "${quantized_smoothquant_path}" ]; then
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

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Benchmark complete. Results saved to: ${LOG_FILE}"
