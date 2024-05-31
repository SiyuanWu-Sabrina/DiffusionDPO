DATASET_NAME=$1

srun -N 1 --gres=gpu:4 --job-name=recap_fid --pty python generate.py $DATASET_NAME