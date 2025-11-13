cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/data1/xiangc/mxy/huggingface
export WANDB_API_KEY=e74ad1b5e2419ca7191227f6ac05268cfb3547de
mkdir -p ./logs

# 打印日志路径以确认
echo "Log path set to: $LOG_PATH"
export CUDA_VISIBLE_DEVICES=0,1

WANDB_MODE=online torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
    --output_dir ./outputs/test_humanomni_emer_1format_withpath_withchoice/ \
    --model_name_or_path /data1/models/EMER-SFT-0.5B \
    --json_dataset_path /data1/xiangc/mxy/R1-Omni/DFEW/dfew_train_fold1.json \
    --data_folder /data1/xiangc/mxy/R1-Omni \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-emotion \
    --save_steps 1000 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  