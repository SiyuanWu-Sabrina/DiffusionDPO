#!/bin/bash
#SBATCH --job-name=all-5e-10
#SBATCH --output=output/all-5e-10.out
#SBATCH --error=output/all-5e-10.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

export MODEL_NAME="/share/Stable-Diffusion/stable-diffusion-xl-base-1.0"
export VAE="/share/Stable-Diffusion/sdxl-vae-fp16-fix"
export DATASET_NAME="all"
export CACHE_DIR="./cache/default/"
export MAX_TRAIN_STEP=256
export WARMUP_STEP=10
export GAS=32
export CHECKPOINTING_STEP=32 # should be a multiple of GAS


# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

# here we use batch size = 256, gradient accumulation steps = 32, 8 GPUs

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAS \
  --max_train_steps=$MAX_TRAIN_STEP \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=$WARMUP_STEP \
  --learning_rate=1e-9 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="all-42-5e-10" \
  --seed 42 \
  --train_data_subdir 0


accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAS \
  --max_train_steps=$MAX_TRAIN_STEP \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=$WARMUP_STEP \
  --learning_rate=1e-9 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="all-167-5e-10" \
  --seed 167 \
  --train_data_subdir 0


accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAS \
  --max_train_steps=$MAX_TRAIN_STEP \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=$WARMUP_STEP \
  --learning_rate=1e-9 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="all-937-5e-10" \
  --seed 937 \
  --train_data_subdir 0