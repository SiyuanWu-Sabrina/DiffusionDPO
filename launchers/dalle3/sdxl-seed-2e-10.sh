#!/bin/bash
#SBATCH --job-name=se-2e10
#SBATCH --output=output/se-2e10.out
#SBATCH --error=output/se-2e10.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

export MODEL_NAME="/share/Stable-Diffusion/stable-diffusion-xl-base-1.0"
export VAE="/share/Stable-Diffusion/sdxl-vae-fp16-fix"
export DATASET_NAME="dalle3"
export CACHE_DIR="./cache/default/"
export MAX_TRAIN_STEP=256
export WARMUP_STEP=10
export GAS=16
export CHECKPOINTING_STEP=16 # should be a multiple of GAS


# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

# here we use batch size = 128, gradient accumulation steps = 16, 8 GPUs

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAS \
  --max_train_steps=$MAX_TRAIN_STEP \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=$WARMUP_STEP \
  --learning_rate=2e-10 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="se-42-2e10" \
  --seed 42

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAS \
  --max_train_steps=$MAX_TRAIN_STEP \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=$WARMUP_STEP \
  --learning_rate=2e-10 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="se-167-2e10" \
  --seed 167

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=$GAS \
  --max_train_steps=$MAX_TRAIN_STEP \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=$WARMUP_STEP \
  --learning_rate=2e-10 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="se-937-2e10" \
  --seed 937
