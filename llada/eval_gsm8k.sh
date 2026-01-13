# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Parse command line arguments
USE_SMOOTHED=false
SMOOTHED_MODEL_PATH="models/llada_smoothquant.pt"

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-smoothed)
            USE_SMOOTHED=true
            shift
            ;;
        --smoothed-model-path)
            SMOOTHED_MODEL_PATH="$2"
            USE_SMOOTHED=true
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--use-smoothed] [--smoothed-model-path PATH]"
            exit 1
            ;;
    esac
done

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'

# Build model_args string
if [ "$USE_SMOOTHED" = true ]; then
    echo "Using SmoothQuant model: $SMOOTHED_MODEL_PATH"
    smoothed_arg="smoothed_model_path=${SMOOTHED_MODEL_PATH}"
else
    smoothed_arg=""
fi


# baseline
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True
fi 


# prefix cache
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True
fi 


# parallel
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True
fi

# parallel factor
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True
fi


# prefix cache+parallel
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True
fi

# dual cache+parallel
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True
fi

# prefix cache+parallel factor
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True
fi

# dual cache+parallel factor
if [ -n "$smoothed_arg" ]; then
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,factor=${factor},show_speed=True,${smoothed_arg}
else
    accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,factor=${factor},show_speed=True
fi
