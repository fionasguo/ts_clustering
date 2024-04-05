#!/bin/bash
#SBATCH --partition=medium
#SBATCH --account=medium
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --mem 0
#SBATCH --time=24:00:00

source ~/anaconda3/envs/damf_env/bin/activate ts_embed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

python setup.py install

# phase 1a data
# python src/classification_train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/incas_data/phase1a_demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/incas_data/phase1a_time_coord_gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_phase1a_bert_pca_lownoise_InfoNCE_temp5 \
#     -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_phase1a_bert_pca_lownoise_InfoNCE_temp5/model_weights.h5

# python src/classification_train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/hamas_data/bert_embeddings_ts_data_3D.pkl \
#     -d /nas/eclairnas01/users/siyiguo/hamas_data/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/hamas_data/gt_data_luca.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_hamas \
#     -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_hamas/model_weights.h5 \
#     -l 650

# python src/classification_train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/hamas_data/bert_embeddings_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/hamas_data/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/hamas_data/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_hamas \
#     -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_hamas/model_weights.h5 \
#     -l 650

# # coord campaign data
# python src/classification_train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/russia/xlmt_embeddings_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/russia/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/russia/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_russia/textual_only \
#     -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_russia/textual_only/model_weights.h5 \
#     -l 1500 \
#     --tau 0.0 \
#     --lam 1.0

# python src/classification_train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/russia/activity_ts_data.pkl \
#     -d /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/russia/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/russia/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_russia/temporal_only \
#     -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_russia/temporal_only/model_weights.h5 \
#     -l 1500 \
#     --tau 1.0 \
#     --lam 1.0

python src/classification_train_and_test.py \
    -m train_test \
    -i /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/avg_embeddings_ts_data.pkl \
    -d /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/demo_data.pkl \
    -g /nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/gt_data.pkl \
    -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_cnhu/textual_only \
    -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_cnhu/textual_only/model_weights.h5 \
    -l 1500 \
    --tau 0.0 \
    --lam 1.0
    

# rvw data
# python src/classification_train_and_test.py \
#     -m train_test \
#     -i /nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_ts_data_autonomy_health.pkl \
#     -d /nas/eclairnas01/users/siyiguo/rvw_data/demo_data_autonomy_health.pkl \
#     -g /nas/eclairnas01/users/siyiguo/rvw_data/gt_data_autonomy_health.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_autonomy_health \
#     -t /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_autonomy_health/model_weights.h5 \
#     -l 610

# python src/classification_train_and_test.py \
#     -m test \
#     -i /nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_ts_data_autonomy_health_0624_1108.pkl \
#     -d /nas/eclairnas01/users/siyiguo/rvw_data/demo_data_autonomy_health.pkl \
#     -g /nas/eclairnas01/users/siyiguo/rvw_data/gt_data_autonomy_health.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_autonomy_health/0624_1108_cla \
#     --trained_classifier_model /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_autonomy_health/classifier_model_weights.h5 \
#     -l 230

# python src/classification_train_and_test.py \
#     -m test \
#     -i /nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_ts_data_0624_1108_3D.pkl \
#     -d /nas/eclairnas01/users/siyiguo/rvw_data/demo_data.pkl \
#     -g /nas/eclairnas01/users/siyiguo/rvw_data/gt_data.pkl \
#     -o /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/textual_only/0624_1108_cla \
#     --trained_classifier_model /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/textual_only/classifier_model_weights.h5 \
#     -l 230 \
#     --tau 0.0 \
#     --lam 1.0
    
    # --trained_classifier_model /nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/textual_only/classifier_model_weights.h5 \
