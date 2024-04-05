#!/bin/bash
#SBATCH --partition=long
#SBATCH --account=long
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem 0
#SBATCH --time=72:00:00

source ~/anaconda3/envs/damf_env/bin/activate ts_embed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

python setup.py install


pip install pandas==1.4
python src/real_test_data_processing/process_rvw_data.py
pip install pandas==2.1.1

# python src/real_test_data_processing/process_keith_coord_camp_data.py

# python src/real_test_data_processing/get_bert_embeddings.py /nas/eclairnas01/users/siyiguo/InfoOpsNationwiseDriverControl/china

# python src/TSCluster/modeling/similarity_search.py "/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/textual_only/0101_0502_cla/"

# python tmp.py
