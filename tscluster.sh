#!/bin/bash
#SBATCH --partition=short
#SBATCH --account=short
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=6:00:00
# #SBATCH --output=test.out

source ~/anaconda3/envs/damf_env/bin/activate ts_embed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

python src/train_and_test.py -m train_test -i data/synthetic_data/synthetic_data_simple_10c_10d_X.pkl -g data/synthetic_data/synthetic_data_simple_10c_10d_y.pkl -o test_simple_10c_10d
