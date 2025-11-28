#!/bin/bash

# 1. Export Environment Variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 2. Set Device Specifics
# Explicitly use the first GPU. 
export CUDA_VISIBLE_DEVICES=0 

# 3. Model Configuration
model_path="Efficient-Large-Model/Fast_dLLM_v2_7B"

# 4. Task Configuration
task="gsm8k"
batch_size=256  # Increased from 32. If you hit OOM (Out of Memory), reduce to 48 or 32.

echo "Starting Fast-dLLM v2 evaluation on ${task} with batch size ${batch_size}..."

# 5. Run Evaluation
# Added --num_processes 1 to ensure single-GPU execution without spawning unnecessary threads.
# Added device_map='auto' to model_args to leverage your 128GB RAM for loading if VRAM is tight.
accelerate launch --num_processes 1 eval.py \
    --tasks ${task} \
    --batch_size ${batch_size} \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args model_path=${model_path},threshold=0.8,show_speed=True,device_map="auto"
