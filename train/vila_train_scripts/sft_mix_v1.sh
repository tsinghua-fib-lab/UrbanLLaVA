#!/bin/bash

# Set the master address to localhost for single node
export MASTER_ADDR="127.0.0.1"
export CURRENT_RANK=0

# Since it's single node, we don't need worker_list or SLURM_JOB_NODELIST
n_node=1

echo "MASTER_ADDR="$MASTER_ADDR
echo "Single node setup, no SLURM required."


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# TODO: Set the output directory
OUTPUT_DIR=""
mkdir $OUTPUT_DIR
# TODO: Set the path to the training script
CODE_PATH=/<path-to-your-dir>/train/VILA/llava/train/train_mem.py
# TODO: Make sure the actural data mixture is correct
DATA_MIX=llava_instruct+sharegpt4v_gpt4_100k+UrbanLLaVA_multi+UrbanLLaVA_single+UrbanLLaVA_text2img2text+UrbanLLaVA_img2text2img+UrbanLLaVA_citywalk_vison
MODEL_MAX_LENGTH=2048
bs=8  # Adjust batch size as needed for your single GPU
echo "number of nodes:" $n_node7n
echo "per device batch size:" $bs
echo "node rank:" $CURRENT_RANK
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

# TODO: Set the path to the model and the vision tower
torchrun --nnodes=$n_node --nproc_per_node=$NUM_GPUS --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    $CODE_PATH \
    --deepspeed ./zero3.json \
    --model_name_or_path /<path-to-your-model>/Llama-3-VILA1.5-8B \
    --version llama_3 \
    --data_mixture $DATA_MIX \
    --vision_tower /<path-to-your-model>/siglip-so400m-patch14-384  \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to tensorboard
