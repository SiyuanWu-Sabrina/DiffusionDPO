#!/bin/bash
#SBATCH --job-name=eval-no-xex-0
#SBATCH --output=result/eval-no-xex-0.out
#SBATCH --error=result/eval-no-xex-0.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

python quick_samples.py no-1e9-0 32 False
python quick_samples.py no-1e9-0 64 False
python quick_samples.py no-1e9-0 128 False
python quick_samples.py no-1e9-0 256 False
python quick_samples.py no-1e9-0 512 False

python quick_samples.py no-2e10-0 32 False
python quick_samples.py no-2e10-0 64 False
python quick_samples.py no-2e10-0 128 False
python quick_samples.py no-2e10-0 256 False
python quick_samples.py no-2e10-0 512 False

python quick_samples.py no-5e10-0 32 False
python quick_samples.py no-5e10-0 64 False
python quick_samples.py no-5e10-0 128 False
python quick_samples.py no-5e10-0 256 False
python quick_samples.py no-5e10-0 512 False