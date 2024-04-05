import os
from datetime import datetime
import pickle
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ztest
import gc

sns.set_style('whitegrid')

data_dir = '/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/'


def create_logger(data_dir):
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=data_dir+logfilename + '.log',
                        format="%(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True

create_logger(data_dir)

# # compaire the distance between ingroup and outgroup
# logging.info('autonomy_health cosine similarity')

# before = pickle.load(open(data_dir+'0101_0502_cla/embedding_similarity_cosine.pkl','rb'))
# before = before.astype('float32')
# after = pickle.load(open(data_dir+'0624_1108_cla/embedding_similarity_cosine.pkl','rb'))
# after = after.astype('float32')
# gt = pickle.load(open('/nas/eclairnas01/users/siyiguo/rvw_data/gt_data.pkl','rb'))
# gt = gt.astype(bool)
# logging.info(f"before shape={before.shape} after shape={after.shape} gt shape={gt.shape}")
# gt = gt[:before.shape[0]]
# right_users = np.where(gt)[0]

# # rand_rows = random.sample(range(before.shape[0]), 100000)
# # before = before[rand_rows,:][:,rand_rows]
# # after = after[rand_rows,:][:,rand_rows]
# # gt = gt[rand_rows]

# x1 = before[~gt,:][:,gt].flatten()
# x2 = after[~gt,:][:,gt].flatten()
# LR_zstats,LR_pval = ztest(x1,x2)
# logging.info(f"\nL <-> R: len={x1.shape[0]}={x2.shape[0]}, before mean={round(np.mean(x1),3)} std={round(np.std(x1),3)}, after mean={round(np.mean(x2),3)} std={round(np.std(x2),3)}, z-test pval={LR_pval}")
# logging.info(f"before median={round(np.median(x1),3)}, after median={round(np.median(x2),3)}")
# logging.info(f"before sum={np.sum(x1)}, after sum={np.sum(x2)}")
# logging.info(f"# dp with increased similarity: {np.sum((x2-x1)>0)}, {round(np.sum((x2-x1)>0)/len(x1)*100,3)}%")
# # plt.figure()
# # sns.kdeplot(x1,label='before')
# # sns.kdeplot(x2,label='after')
# # plt.legend()
# # plt.savefig(data_dir+'compare_dist_LR.png')

# before[np.triu_indices(len(before),k=0)] = -100.0
# after[np.triu_indices(len(after),k=0)] = -100.0

# x1 = before[~gt,:][:,~gt].flatten()
# x1 = x1[x1!=-100.0]
# x2 = after[~gt,:][:,~gt].flatten()
# x2 = x2[x2!=-100.0]
# LL_zstats,LL_pval = ztest(x1,x2)
# logging.info(f"\nL <-> L: len={x1.shape[0]}={x2.shape[0]}, before mean={round(np.mean(x1),3)} std={round(np.std(x1),3)}, after mean={round(np.mean(x2),3)} std={round(np.std(x2),3)}, z-test pval={LR_pval}")
# logging.info(f"before median={round(np.median(x1),3)}, after median={round(np.median(x2),3)}")
# logging.info(f"before sum={np.sum(x1)}, after sum={np.sum(x2)}")
# logging.info(f"# dp with increased similarity: {np.sum((x2-x1)>0)}, {round(np.sum((x2-x1)>0)/len(x1)*100,3)}%")
# # plt.figure()
# # sns.kdeplot(x1,label='before')
# # sns.kdeplot(x2,label='after')
# # plt.legend()
# # plt.savefig(data_dir+'compare_dist_LL.png')

# x1 = before[gt,:][:,gt].flatten()
# x1 = x1[x1!=-100.0]
# x2 = after[gt,:][:,gt].flatten()
# x2 = x2[x2!=-100.0]
# RR_zstats,RR_pval = ztest(x1,x2)
# logging.info(f"\nR <-> R: len={x1.shape[0]}={x2.shape[0]}, before mean={round(np.mean(x1),3)} std={round(np.std(x1),3)}, after mean={round(np.mean(x2),3)} std={round(np.std(x2),3)}, z-test pval={LR_pval}")
# logging.info(f"before median={round(np.median(x1),3)}, after median={round(np.median(x2),3)}")
# logging.info(f"before sum={np.sum(x1)}, after sum={np.sum(x2)}")
# logging.info(f"# dp with increased similarity: {np.sum((x2-x1)>0)}, {round(np.sum((x2-x1)>0)/len(x1)*100,3)}%")
# # plt.figure()
# # sns.kdeplot(x1,label='before')
# # sns.kdeplot(x2,label='after')
# # plt.legend()
# # plt.savefig(data_dir+'compare_dist_RR.png')

# KNN

gt = pickle.load(open('/nas/eclairnas01/users/siyiguo/rvw_data/gt_data.pkl','rb'))
gt = gt.astype(bool)
# gt = gt[:p_right_neighbors_before.shape[0]]
right_users = np.where(gt)[0]

def helper(before_path,K,right_users):
    before = pickle.load(open(data_dir+before_path+'/embedding_similarity_cosine.pkl','rb'))
    before = before.astype('float32')
    before = np.argpartition(-before,K,axis=1)[:,:K]
    p_right_neighbors_before = np.sum(np.isin(before,right_users),axis=1)/K
    p_left_neighbors_before = 1 - p_right_neighbors_before

    del before
    gc.collect()

    return p_right_neighbors_before,p_left_neighbors_before

def knn(K,right_users,gt):
    p_right_neighbors_before, p_left_neighbors_before = helper('0101_0502_cla',K,right_users)
    p_right_neighbors_after, p_left_neighbors_after = helper('0624_1108_cla',K,right_users)
    logging.info(f"\nnum right users={len(right_users)}, p_right_neighbors_before shape={p_right_neighbors_before.shape}\n{p_right_neighbors_before[:10]}")

    gt = gt[:p_right_neighbors_before.shape[0]]

    logging.info(f"\nkNN with K={K}:")
    logging.info(f"L - L neighbors before={round(np.mean(p_left_neighbors_before[~gt]),3)}, {round(np.std(p_left_neighbors_before[~gt]),3)}, after={round(np.mean(p_left_neighbors_after[~gt]),3)}, {round(np.std(p_left_neighbors_after[~gt]),3)}, z-test pval={ztest(p_left_neighbors_before[~gt],p_left_neighbors_after[~gt])[1]}")
    logging.info(f"L - R neighbors before={round(np.mean(p_right_neighbors_before[~gt]),3)}, {round(np.std(p_right_neighbors_before[~gt]),3)}, after={round(np.mean(p_right_neighbors_after[~gt]),3)}, {round(np.std(p_right_neighbors_after[~gt]),3)}, z-test pval={ztest(p_right_neighbors_before[~gt],p_right_neighbors_after[~gt])[1]}")
    logging.info(f"R - L neighbors before={round(np.mean(p_left_neighbors_before[gt]),3)}, {round(np.std(p_left_neighbors_before[gt]),3)}, after={round(np.mean(p_left_neighbors_after[gt]),3)}, {round(np.std(p_left_neighbors_after[gt]),3)}, z-test pval={ztest(p_left_neighbors_before[gt],p_left_neighbors_after[gt])[1]}")
    logging.info(f"R - R neighbors before={round(np.mean(p_right_neighbors_before[gt]),3)}, {round(np.std(p_right_neighbors_before[gt]),3)}, after={round(np.mean(p_right_neighbors_after[gt]),3)}, {round(np.std(p_right_neighbors_after[gt]),3)}, z-test pval={ztest(p_right_neighbors_before[gt],p_right_neighbors_after[gt])[1]}")

for k in [20,200,500,1000]:
    knn(k,right_users,gt)

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, verbose=0)
# tsne_results = tsne.fit_transform(feats)

# plt.figure(figsize=(8, 8))
# # plt.xlim((-40, 40))
# # plt.ylim((-40, 40))
# fig = sns.scatterplot(x=tsne_results[:, 0],
#                         y=tsne_results[:, 1],
#                         hue=labels,
#                         palette=sns.color_palette(
#                             "hls", len(np.unique(labels))),
#                         legend="full",
#                         alpha=0.3).get_figure()

# fig.savefig('/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D/org_data_tsne.png', format='png')