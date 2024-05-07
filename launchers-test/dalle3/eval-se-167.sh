#!/bin/bash
#SBATCH --job-name=eval-se-167-xex
#SBATCH --output=result/eval-se-167-xex.out
#SBATCH --error=result/eval-se-167-xex.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

python quick_samples.py se-167-1e9 16 False
python quick_samples.py se-167-1e9 32 False
python quick_samples.py se-167-1e9 64 False
python quick_samples.py se-167-1e9 128 False
python quick_samples.py se-167-1e9 256 False
python quick_samples.py se-167-1e9 512 False

python quick_samples.py se-167-2e10 16 False
python quick_samples.py se-167-2e10 32 False
python quick_samples.py se-167-2e10 64 False
python quick_samples.py se-167-2e10 128 False
python quick_samples.py se-167-2e10 256 False
python quick_samples.py se-167-2e10 512 False

python quick_samples.py se-167-5e10 16 False
python quick_samples.py se-167-5e10 32 False
python quick_samples.py se-167-5e10 64 False
python quick_samples.py se-167-5e10 128 False
python quick_samples.py se-167-5e10 256 False
python quick_samples.py se-167-5e10 512 False
