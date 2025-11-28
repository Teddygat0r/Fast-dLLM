# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path=Efficient-Large-Model/Fast_dLLM_v2_7B

# --- Performance Flags for DGX Spark (Single GPU, BF16, Inductor) ---
ACCEL_FLAGS="--num_processes 1 --mixed_precision bf16 --dynamo_backend inductor"
# ---------------------------------------------------------------------

# Task: mmlu
task=mmlu
accelerate launch ${ACCEL_FLAGS} eval.py --tasks ${task} --batch_size 1 --num_fewshot 5 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}

# Task: gpqa_main_n_shot
task=gpqa_main_n_shot
accelerate launch ${ACCEL_FLAGS} eval.py --tasks ${task} --batch_size 1 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path}

# Task: gsm8k
task=gsm8k
accelerate launch ${ACCEL_FLAGS} eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True

# Task: minerva_math
task=minerva_math
accelerate launch ${ACCEL_FLAGS} eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True

# Task: ifeval
task=ifeval
accelerate launch ${ACCEL_FLAGS} eval.py --tasks ${task} --batch_size 32 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True
