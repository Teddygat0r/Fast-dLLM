#!/bin/bash

# Speed comparison benchmark for different model configurations.
# Tests generation speed (tokens/second) across base, SmoothQuant, and quantized models.
# Uses a limited number of samples (50) for faster benchmarking.

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
limit=50  # Limit to 50 samples for faster benchmarking
model_path="GSAI-ML/LLaDA-8B-Instruct"
smoothed_model_path="models/llada_smoothquant.pt"
quantized_model_path="models/llada_quantized_w4a8_full.pt"
quantized_smoothquant_path="models/llada_quantized_w4a8_smoothquant_full.pt"

# Prepare log file
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/speed_comparison_${TIMESTAMP}.log"

{
  echo "============================================================"
  echo "Speed Comparison Benchmark (${limit} samples)"
  echo "============================================================"
  echo "Task:          ${task}"
  echo "Model path:    ${model_path}"
  echo "Gen length:    ${length}"
  echo "Block length:  ${block_length}"
  echo "Num few-shot:  ${num_fewshot}"
  echo "Limit:         ${limit} samples"
  echo "Prefix cache:  use_cache=True"
  echo "Show speed:    show_speed=True"
  echo "Timestamp:     ${TIMESTAMP}"
  echo "============================================================"
  echo ""

  echo "=== 1) Base model (prefix cache + parallel) ==="
  accelerate launch eval_llada.py \
    --tasks "${task}" \
    --num_fewshot "${num_fewshot}" \
    --limit "${limit}" \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --model_args "model_path=${model_path},gen_length=${length},steps=$((length / block_length)),block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True"

  echo ""
  echo "=== 2) SmoothQuant model (prefix cache + parallel) ==="
  if [ -f "${smoothed_model_path}" ]; then
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --limit "${limit}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=$((length / block_length)),block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True,smoothed_model_path=${smoothed_model_path}"
  else
    echo "  WARNING: SmoothQuant model not found at ${smoothed_model_path}"
    echo "  Skipping SmoothQuant benchmark."
  fi

  echo ""
  echo "=== 3) Quantized model W4A8 (prefix cache + parallel) ==="
  if [ -f "${quantized_model_path}" ]; then
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --limit "${limit}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=$((length / block_length)),block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True,quantized_model_path=${quantized_model_path}"
  else
    echo "  WARNING: Quantized model not found at ${quantized_model_path}"
    echo "  Skipping quantized benchmark."
  fi

  echo ""
  echo "=== 4) Quantized + SmoothQuant W4A8 (prefix cache + parallel) ==="
  if [ -f "${quantized_smoothquant_path}" ]; then
    accelerate launch eval_llada.py \
      --tasks "${task}" \
      --num_fewshot "${num_fewshot}" \
      --limit "${limit}" \
      --confirm_run_unsafe_code \
      --model llada_dist \
      --model_args "model_path=${model_path},gen_length=${length},steps=$((length / block_length)),block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True,quantized_model_path=${quantized_smoothquant_path}"
  else
    echo "  WARNING: Quantized+SmoothQuant model not found at ${quantized_smoothquant_path}"
    echo "  Skipping quantized+SmoothQuant benchmark."
  fi

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Speed comparison complete. Results saved to: ${LOG_FILE}"
echo ""
echo "Summary: Compare 'Tokens per second' values across different configurations."
