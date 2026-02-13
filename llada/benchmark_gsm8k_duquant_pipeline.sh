#!/bin/bash

# Benchmark GSM8k using the DuQuant pipeline as in eval_llada.py.
# Loads a pre-saved DuQuant model via duquant_weight_path (QuantLinear layers
# from model/quantize, config from model/quantize/quant_args.json).
#
# Uses prefix caching (use_cache=True) and show_speed=True, and writes
# all console output to a timestamped log file in ./logs.
#
# Usage:
#   bash benchmark_gsm8k_duquant_pipeline.sh                    # run all benchmarks (default)
#   bash benchmark_gsm8k_duquant_pipeline.sh --only base       # only base model (no quantization)
#   bash benchmark_gsm8k_duquant_pipeline.sh --only duquant    # only DuQuant (requires weight path)
#   bash benchmark_gsm8k_duquant_pipeline.sh --limit 10        # limit number of GSM8k questions
#   bash benchmark_gsm8k_duquant_pipeline.sh --batch-size 4    # set batch size
#   bash benchmark_gsm8k_duquant_pipeline.sh --duquant-weight-path path/to/model.pth

set -euo pipefail

# Move to the directory containing this script so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Environment needed by eval_llada.py / HF datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Restrict to a single GPU
export CUDA_VISIBLE_DEVICES=1

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
wbits=8
abits=8
symmetric=false
model_path="GSAI-ML/LLaDA-8B-Instruct"

# Prefix cache setting
use_prefix_cache=true

# DuQuant: path to pre-saved DuQuant parameters (.pth) as used in eval_llada.py.
# Quant config is read from model/quantize/quant_args.json.
# Set to empty "" to skip the DuQuant benchmark (no on-the-fly calibration in eval_llada).
duquant_weight_path="models/quantized_model.pth"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  echo "Benchmark GSM8k with DuQuant Pipeline (matches eval_llada.py)"
  echo ""
  echo "Usage:"
  echo "  $0                           # run all benchmarks"
  echo "  $0 --only base               # only base model (no quantization)"
  echo "  $0 --only duquant            # only DuQuant pipeline (requires --duquant-weight-path)"
  echo ""
  echo "Options:"
  echo "  --limit N                    # limit to N GSM8k questions (default: full dataset)"
  echo "  --batch-size N               # batch size for evaluation (default: 1)"
  echo "  --wbits N                    # weight bits for DuQuant (default: 8)"
  echo "  --abits N                    # activation bits for DuQuant (default: 8)"
  echo "  --duquant-weight-path PATH   # path to pre-saved DuQuant .pth (default: models/quantized_model.pth)"
  echo "  --no-duquant-weights         # skip DuQuant benchmark (do not run with saved weights)"
  echo "  --no-prefix-cache            # disable prefix cache (use_cache=False)"
  echo "  --symmetric                  # use symmetric quantization (default: False)"
  echo "  -h, --help                   # show this help"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || usage
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
    --duquant-weight-path)
      [[ $# -ge 2 ]] || usage
      duquant_weight_path="$2"
      shift 2
      ;;
    --no-duquant-weights)
      duquant_weight_path=""
      shift
      ;;
    --no-prefix-cache)
      use_prefix_cache=false
      shift
      ;;
    --wbits)
      [[ $# -ge 2 ]] || usage
      wbits="$2"
      shift 2
      ;;
    --abits)
      [[ $# -ge 2 ]] || usage
      abits="$2"
      shift 2
      ;;
    --symmetric)
      [[ $# -ge 2 ]] || usage
      symmetric="$2"
      shift 2
      ;;
    -h)
      usage
      ;;
    --help)
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
  echo "GSM8k Benchmark: DuQuant Pipeline (eval_llada.py)"
  echo "============================================================"
  echo "Task:              ${task}"
  echo "Model path:        ${model_path}"
  echo "Gen length:        ${length}"
  echo "Block length:      ${block_length}"
  echo "Batch size:        ${batch_size}"
  echo "Num few-shot:      ${num_fewshot}"
  echo "W bits:            ${wbits}"
  echo "A bits:            ${abits}"
  echo "Symmetric:         ${symmetric}"
  echo "Limit:             ${limit:-'full dataset'}"
  echo "Prefix cache:      use_cache=${use_prefix_cache^}"
  echo "Show speed:        show_speed=True"
  echo ""
  echo "DuQuant (eval_llada.py):"
  echo "  Weight path:     ${duquant_weight_path:-'(none – DuQuant run skipped)'}"
  echo "  Config:          model/quantize/quant_args.json"
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
    if [[ -z "${duquant_weight_path}" ]]; then
      echo "=== 2) DuQuant Pipeline (skipped: no weight path) ==="
      echo "  Provide --duquant-weight-path to run DuQuant benchmark."
      echo ""
    elif [[ ! -f "${duquant_weight_path}" ]]; then
      echo "=== 2) DuQuant Pipeline (skipped: file not found) ==="
      echo "  Weight path: ${duquant_weight_path}"
      echo "  Create the file or pass --duquant-weight-path to an existing .pth."
      echo ""
    else
      echo "=== 2) DuQuant Pipeline with prefix cache ==="
      echo "  Loading pre-saved DuQuant model from: ${duquant_weight_path}"
      accelerate launch eval_llada.py \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        ${limit_arg} \
        --batch_size "${batch_size}" \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=${use_prefix_cache^},show_speed=True,use_duquant_pipeline=True,duquant_weight_path=${duquant_weight_path},wbits=${wbits},abits=${abits},symmetric=${symmetric}"
      echo ""
    fi
  else
    echo "Skipping DuQuant pipeline benchmark (per arguments)."
    echo ""
  fi

} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Benchmark complete. Results saved to: ${LOG_FILE}"
