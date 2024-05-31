#!/bin/bash
#SBATCH --job-name=gen_dalle3
#SBATCH --output=result/gen_dalle3.out
#SBATCH --error=result/gen_dalle3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

python quick_samples.py dalle3-167-1e-9 32 True
python quick_samples.py dalle3-42-1e-9 32 True
python quick_samples.py dalle3-42-1e-9 256 True
