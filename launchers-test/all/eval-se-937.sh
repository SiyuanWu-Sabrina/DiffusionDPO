#!/bin/bash
#SBATCH --job-name=eval-all-937-xex
#SBATCH --output=result/eval-all-937-xex.out
#SBATCH --error=result/eval-all-937-xex.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=saber_wu@126.com

# python quick_samples.py all-937-1e-9 32 False
# python quick_samples.py all-937-1e-9 64 False
# python quick_samples.py all-937-1e-9 128 False
# python quick_samples.py all-937-1e-9 256 False
# python quick_samples.py all-937-1e-9 512 False

# python quick_samples.py all-937-2e-10 32 False
# python quick_samples.py all-937-2e-10 64 False
# python quick_samples.py all-937-2e-10 128 False
# python quick_samples.py all-937-2e-10 256 False
# python quick_samples.py all-937-2e-10 512 False

# python quick_samples.py all-937-5e-10 32 False
# python quick_samples.py all-937-5e-10 64 False
# python quick_samples.py all-937-5e-10 128 False
# python quick_samples.py all-937-5e-10 256 False
# python quick_samples.py all-937-5e-10 512 False

python quick_samples.py all-937-1e-9 1024 False
python quick_samples.py all-937-1e-9 2048 False

python quick_samples.py all-937-2e-10 1024 False
python quick_samples.py all-937-2e-10 2048 False

python quick_samples.py all-937-5e-10 1024 False
python quick_samples.py all-937-5e-10 2048 False
