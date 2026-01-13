#!/bin/bash

# Benchmark GSM8k on the base model and then the SmoothQuant model.
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

# Prepare log file
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/gsm8k_base_vs_smoothquant_${TIMESTAMP}.log"

{
  echo "============================================================"
  echo "GSM8k Benchmark: Base vs SmoothQuant (prefix cache enabled)"
  echo "============================================================"
  echo "Task:          ${task}"
  echo "Model path:    ${model_path}"
  echo "SmoothQuant:   ${smoothed_model_path}"
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
  accelerate launch eval_llada.py \
    --tasks "${task}" \
    --num_fewshot "${num_fewshot}" \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,smoothed_model_path=${smoothed_model_path}"

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Benchmark complete. Results saved to: ${LOG_FILE}"

