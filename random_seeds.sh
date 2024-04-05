#!/bin/bash
#SBATCH --partition=medium-lg
#SBATCH --account=medium-lg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=24:00:00

source ~/anaconda3/envs/damf_env/bin/activate ts_embed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

python setup.py install

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    echo "################################### seed $SEED ###################################"
    python src/classification_train_and_test.py \
        -m train_test \
        -i /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/xlmt_embeddings_ts_data.pkl \
        -d /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/demo_data.pkl \
        -g /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/gt_data.pkl \
        -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_cnhu \
        -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_cnhu/model_weights.h5 \
        -l 1500 \
        -s $SEED
done