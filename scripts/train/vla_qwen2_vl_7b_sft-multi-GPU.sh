epoch=1
batch=16
gradient_accumulation_steps=1
card_number=1
cuda_visible_devices=0,1,2,3,4,5,6,7
node_number=1
total_number=$(($card_number * $node_number))
training_port=24001

dataset_name="CraftJarvis/JarvisVLA-Qwen2-VL-7B" #end before '-train.json' and '-valid.json'
base_model_path="/share/models/mc-base-qwen2-vl-7b-2502"
version="mc-vlp-vla-qwen2-vl-7b"
WANDB_NAME="$version-c$total_number-e$epoch-b$batch-a$gradient_accumulation_steps"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port \
    jarvisvla/train/train.py \
    --deepspeed configs/deepspeed_config_s1.json \
    --dataset_name $dataset_name \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --seed 43 \
    --model_name_or_path $base_model_path \
    --report_to "wandb" \
    --learning_rate 5e-6 \
    --max_grad_norm 1 \
    --weight_decay 0. \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 1600 \
    --save_total_limit 5 \
    --output_dir "/public/JARVIS/checkpoints2/$WANDB_NAME" \
    --run_name $WANDB_NAME \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 512 \
    --collator_type "VLAMultimodalChatDataCollatorforVLM" \
    --fix_visual_encoder True \
    --fix_visual_adapter True \

    