#!/bin/bash
#SBATCH --job-name=gen_sd
#SBATCH --output=result/gen_sd.out
#SBATCH --error=result/gen_sd.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

python quick_samples.py stable_diffusion-937-1e-9 64 True
python quick_samples.py stable_diffusion-42-1e-9 32 True
python quick_samples.py stable_diffusion-167-1e-9 256 True
