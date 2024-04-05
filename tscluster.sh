#!/bin/bash
#SBATCH --partition=medium-lg
#SBATCH --account=medium-lg
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=24:00:00
# #SBATCH --output=test.out
# #SBATCH --gpus=1
source ~/anaconda3/envs/damf_env/bin/activate ts_embed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

python setup.py install

# # phase 1a data
# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/incas_data/phase1a_demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/incas_data/phase1a_time_coord_gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_phase1a_bert_pca_lownoise_InfoNCE_temp5 \
#     -n 0.05 \
#     -l 1000 \
#     --temperature 5

# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/hamas_data/bert_embeddings_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/hamas_data/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/hamas_data/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_hamas \
#     -n 3 \
#     -l 650 \
#     --temperature 0.8

# # ED data
# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/eating_disorder_data/bert_embeddings_ts_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/eating_disorder_data/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_ED_bertpca_highnoise_SimSiam \
#     -n 1 \
#     -l 1000

# Luca's coord camp data
# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/uae/activity_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/uae/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/uae/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_uae/temporal_only \
#     -n 2 \
#     -l 2500 \
#     --temperature 3 \
#     --tau 1.0 \
#     --lam 1.0

# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/activity_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_cnhu/temporal_only \
#     -n 2 \
#     -l 1200 \
#     --temperature 3 \
#     --tau 1.0 \
#     --lam 1.0

# max triplet len: egyptuae=2200; venezuela=2700; cnhu=1200

# RVW data
# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_ts_data_autonomy_health.pkl \
#     -d /nas/eclairnas01/users/siyiguo/rvw_data/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/rvw_data/gt_data_autonomy_health.pkl \
#     -d /nas/eclairnas01/users/siyiguo/rvw_data/demo_data_autonomy_health.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_autonomy_health \
#     -n 2 \
#     -l 610 \
#     --temperature 3


python src/train_and_test.py \
    -m train_test \
    -i /nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_emotmf_ts_data_3D.pkl \
    -d /nas/eclairnas01/users/siyiguo/rvw_data/demo_data.pkl \
    -g /nas/eclairnas01/users/siyiguo/rvw_data/gt_data.pkl \
    -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo_emotmf \
    -n 2 \
    -l 3000 \
    --temperature 3 \
    --tau 1.0 \
    --lam 1.0

## synthetic data
# python src/train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/ts_clustering/data/synthetic_data/same_content_diff_activity_X.pkl \
#     -g /nas/eclairnas01/users/siyiguo/ts_clustering/data/synthetic_data/same_content_diff_activity_y.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_same_content_diff_activity \
#     -n 0.5 \
#     -l 500


