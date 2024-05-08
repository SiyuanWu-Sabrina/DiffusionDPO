#!/bin/bash
#SBATCH --job-name=eval-dalle3-42-xex
#SBATCH --output=result/eval-dalle3-42-xex.out
#SBATCH --error=result/eval-dalle3-42-xex.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

python quick_samples.py dalle3-42-1e-9 32 False
python quick_samples.py dalle3-42-1e-9 64 False
python quick_samples.py dalle3-42-1e-9 128 False
python quick_samples.py dalle3-42-1e-9 256 False
python quick_samples.py dalle3-42-1e-9 512 False

python quick_samples.py dalle3-42-2e-10 32 False
python quick_samples.py dalle3-42-2e-10 64 False
python quick_samples.py dalle3-42-2e-10 128 False
python quick_samples.py dalle3-42-2e-10 256 False
python quick_samples.py dalle3-42-2e-10 512 False

python quick_samples.py dalle3-42-5e-10 32 False
python quick_samples.py dalle3-42-5e-10 64 False
python quick_samples.py dalle3-42-5e-10 128 False
python quick_samples.py dalle3-42-5e-10 256 False
python quick_samples.py dalle3-42-5e-10 512 False
