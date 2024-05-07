#!/bin/bash
#SBATCH --job-name=no-5e10-0
#SBATCH --output=output/no-5e10-0.out
#SBATCH --error=output/no-5e10-0.err
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
export MAX_TRAIN_STEP=512
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
  --learning_rate=5e-10 --scale_lr \
  --cache_dir=$CACHE_DIR \
  --checkpointing_steps $CHECKPOINTING_STEP \
  --beta_dpo 5000 \
   --sdxl  \
  --caption_csv_file /share/imagereward_work/prompt_reconstruction/data/blip2_flan.csv \
  --output_dir="no-5e10-0" \
  --train_data_subdir 0
