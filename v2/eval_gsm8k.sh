#!/bin/bash

# Parse command-line arguments
USE_SMOOTHQUANT=false
LIMIT=""
BATCH_SIZE=256
THRESHOLD=0.9

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoothquant|-sq)
            USE_SMOOTHQUANT=true
            shift
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --threshold|-t)
            THRESHOLD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoothquant, -sq        Use SmoothQuant model (default: base model)"
            echo "  --limit, -l NUM          Limit number of samples (e.g., -l 20)"
            echo "  --batch-size, -b NUM     Batch size (default: 256)"
            echo "  --threshold, -t NUM      Generation threshold (default: 0.9)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --smoothquant --limit 20"
            echo "  $0 -sq -l 20 -b 8 -t 0.9"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 1. Export Environment Variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 2. Set Device Specifics
# Explicitly use the first GPU. 
export CUDA_VISIBLE_DEVICES=0 

# 3. Model Configuration
model_path="Efficient-Large-Model/Fast_dLLM_v2_7B"
smoothquant_model_path="models/fast_dllm_smoothquant.pt"

# 4. Task Configuration
task="gsm8k"

echo "Starting Fast-dLLM v2 evaluation on ${task} with batch size ${BATCH_SIZE}..."

# Check if SmoothQuant model should be used
if [ "$USE_SMOOTHQUANT" = true ]; then
    if [ -f "$smoothquant_model_path" ]; then
        echo "Using SmoothQuant model: ${smoothquant_model_path}"
        model_args="model_path=${model_path},smoothquant_model_path=${smoothquant_model_path},threshold=${THRESHOLD},show_speed=True,device_map=auto,load_in_8bit=True"
    else
        echo "ERROR: SmoothQuant model not found: ${smoothquant_model_path}"
        echo "Falling back to base model..."
        model_args="model_path=${model_path},threshold=${THRESHOLD},show_speed=True,device_map=auto,load_in_8bit=True"
    fi
else
    echo "Using base model (no SmoothQuant)"
    model_args="model_path=${model_path},threshold=${THRESHOLD},show_speed=True,device_map=auto,load_in_8bit=True"
fi

echo "Generation threshold: ${THRESHOLD}"

# 5. Run Evaluation
# Added --num_processes 1 to ensure single-GPU execution without spawning unnecessary threads.
# Added device_map='auto' to model_args to leverage your 128GB RAM for loading if VRAM is tight.
eval_cmd="accelerate launch --num_processes 1 eval.py \
    --tasks ${task} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args '${model_args}'"

# Add --limit if specified
if [ -n "$LIMIT" ]; then
    eval_cmd="${eval_cmd} --limit ${LIMIT}"
    echo "Limiting evaluation to ${LIMIT} samples"
fi

# Execute the command
eval $eval_cmd
