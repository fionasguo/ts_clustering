import os, sys
from datetime import datetime
import logging
import pickle
import numpy as np
import random
import math
import faiss
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_logger():
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True


def similarity_search(embeddings,save_dir,metric='euclidean'):
    """
    Perform Faiss simialrity search based on the embeddings
    embeddings: np.array N*D, or list of lists
    """
    logging.info(f'Start Faiss simialrity search with {metric} metric')

    embeddings = np.array(embeddings,dtype='float32')
    N,D = embeddings.shape

    if metric == 'euclidean':
        index = faiss.IndexFlatL2(D)
    elif metric == 'cosine':
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(D)
    else:
        raise ValueError('please choose from similarity search metric: euclidean, cosine')

    index.train(embeddings)
    index.add(embeddings)
    logging.info('finished index train and add')

    logging.info(f'start computing distances, N={N}')
    all_distances = np.empty((N,0),dtype='float32')

    n_blocks = 4
    k = math.ceil(N/n_blocks)
    for i in tqdm(range(n_blocks)):
        k_i = N - k*(n_blocks-1) if i == n_blocks-1 else k
        distances = np.empty((N,k_i),dtype='float32')
        labels = np.array([list(range(i*k,(i*k+k_i))) for _ in range(N)],dtype='int64')

        index.compute_distance_subset(N, faiss.swig_ptr(embeddings), k_i, faiss.swig_ptr(distances), faiss.swig_ptr(labels))
        
        all_distances = np.hstack((all_distances,distances))

    logging.info(f'finished computing distances, size of dist matrix {all_distances.shape}')

    if metric == 'euclidean':
        all_distances = 1 - (all_distances - np.min(all_distances)) / (np.max(all_distances) - np.min(all_distances))

    pickle.dump(all_distances,open(save_dir+'embedding_similarity_'+metric+'.pkl','wb'))

    return all_distances

def plot_network(adjacency_matrix,save_dir):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr)#, node_size=500, labels=mylabels, with_labels=True)
    plt.savefig(save_dir+'high_similarity_user_network.png')

    pickle.dump(gr,open(save_dir+'high_similarity_user_network.pkl','wb'))

    return gr


if __name__ == "__main__":
    create_logger()

    data_dir = sys.argv[1]
    # data_dir = '/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/0101_0502_temporal_only_cla'
    logging.info(data_dir)

    # rows_id = pickle.load(open('/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/autonomy_health_0101_0502_cla/classifier_test_embeddings_random_row_idx.pkl','rb'))
    embeddings = pickle.load(open(data_dir+'classifier_test_embeddings.pkl','rb'))
    embeddings = embeddings.astype('float32')
    # rows_id = random.sample(range(embeddings.shape[0]), int(0.6*embeddings.shape[0]))
    # logging.info(f"total N={embeddings.shape[0]} randomly sample {len(rows_id)} rows")
    # pickle.dump(rows_id,open('/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/autonomy_health_0101_0502_cla/classifier_test_embeddings_random_row_idx.pkl','wb'))
    # embeddings = embeddings[rows_id,:]

    sim_mtx = similarity_search(embeddings,data_dir,metric='cosine')

    # ## spearman r corr between time_coord_gt adjacency matrix and the cosine similarity matrix
    # import scipy
    # pred_sim_mtx = pickle.load(open('/nas/eclairnas01/users/siyiguo/ts_clustering/test_rvw_3D_demo/0101_0502_cla/classifier_test_embeddings.pkl','rb'))
    # gt_adj_mtx = pickle.load(open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_time_coord_gt_data_adj_mtx.pkl','rb'))
    # spearman_corr = scipy.stats.spearmanr(gt_adj_mtx,pred_sim_mtx,axis=None)
    # logging.info(spearman_corr)


    ## plot network using networkx - doesn't work well
    # sim_mtx = pickle.load(open('/nas/home/siyiguo/ts_clustering/test_phase1a_bert_pca_multihighnoise/embedding_similarity_cosine.pkl','rb'))
    # sim_mtx[np.triu_indices(len(sim_mtx),k=0)] = -1 # lower triangular mtx

    # all_sim_vals = sim_mtx.flatten()
    # all_sim_vals = all_sim_vals[all_sim_vals>-1]
    # high_sim_val = np.quantile(all_sim_vals, 0.85)
    # logging.info(f"size of all sim values: {len(all_sim_vals)}, 0.9 quantile of simialrity: {high_sim_val}")

    # adj_mtx = (sim_mtx>high_sim_val).astype(int)
    # plot_network(adj_mtx,'/nas/home/siyiguo/ts_clustering/test_phase1a_bert_pca_multihighnoise/')

    # logging.info('finished plotting and saving network')