#!/bin/bash
#SBATCH --job-name=gen_laion
#SBATCH --output=result/gen_laion.out
#SBATCH --error=result/gen_laion.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

python quick_samples.py laion_high_res-937-1e-9 128 True
python quick_samples.py laion_high_res-937-1e-9 64 True
python quick_samples.py laion_high_res-167-1e-9 256 True
