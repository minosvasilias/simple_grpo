#!/bin/bash
set -e

echo "=== Starting the training run ==="

# Activate the virtual environment (if not already active)
source venv/bin/activate

# Store the timestamp once for consistent use
ts=$(date +%Y%m%d_%H%M%S)

# Launch training using Accelerate with the zero3 configuration
nohup accelerate launch --num_processes 3 --config_file configs/zero3.yaml src/train_gsm8k.py \
    --output_dir outputs/Llama-3.1-8B-Instruct.GRPO \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 20 \
    --learning_rate 3e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --num_generations 3 \
    --save_steps 50 \
    --max_steps 1000 \
    --torch_dtype bfloat16 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --bf16 \
    > "train_logs_${ts}.out" 2>&1 < /dev/null &

echo "=== Training run started! Run 'tail -f train_logs_${ts}.out' to monitor. ==="
