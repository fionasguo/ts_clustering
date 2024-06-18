import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import math
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN,MiniBatchKMeans
from sklearn.metrics import classification_report
import pickle
import tensorflow as tf

from .model import SimSiam
# from .clustering import kmeans_clustering


def compute_embedding(data, simsiam, args):
    """
    data: pair (te_data,te_aug_data)
    """
    embeddings = simsiam.encoder.predict(data,batch_size=args['batch_size'])
    return embeddings


def plot_tsne(feats, labels, fig_save_path: str):
    """
    Plot feature embeddings as tsne.

    Can be used to see if source and target domain have distinct features.
    """
    tsne = TSNE(n_components=2, verbose=0)
    tsne_results = tsne.fit_transform(feats)
    
    plt.figure(figsize=(8, 8))
    # plt.xlim((-40, 40))
    # plt.ylim((-40, 40))
    fig = sns.scatterplot(x=tsne_results[:, 0],
                          y=tsne_results[:, 1],
                          hue=labels,
                          palette=sns.color_palette(
                              "hls", len(np.unique(labels))),
                          legend="full",
                          alpha=0.3).get_figure()

    # plt.scatter(tsne_results[labels==0,0],tsne_results[labels==0,1],color='lightskyblue',alpha=0.5,s=3,label='Liberal')
    # plt.scatter(tsne_results[labels==1,0],tsne_results[labels==1,1],color='lightcoral',alpha=0.3,s=3,label='Conservative')
    # plt.legend(ncol=2)

    plt.savefig(fig_save_path + 'tsne.png', format='png')

########################################################################################

def make_combinations(array,batch_size):
    size = tf.shape(array)[0]
    # Make 2D grid of indices
    r = tf.range(size)
    ii, jj = tf.meshgrid(r, r, indexing='ij')
    # Take pairs of indices where the first is less or equal than the second
    m = ii < jj
    m.set_shape([batch_size,batch_size])
    output_size = int((batch_size-1)/2*batch_size)
    return tf.gather(array, tf.reshape(tf.boolean_mask(ii, m),[output_size])),tf.gather(array,tf.reshape(tf.boolean_mask(jj, m),[output_size]))

def prepare_link_data(emb,indices,links,batch_size):
    """
    make pair combinations of all rows in emb, and label them with 1 if there's a link exists between them

    emb: tensor batch_size * hidden_dim
    indices: the index of each emb, (batch_size,)
    links: for each emb, the indicies of other embs that have a link with, batch_sizw * max # links (padded with -1)
    """
    emb1, emb2 = make_combinations(emb,batch_size)
    idx, tgt_idx = make_combinations(indices,batch_size)
    idx_to_links = tf.where(tf.equal(indices,idx[...,None]))[:,-1]
    expanded_links = tf.gather(links,indices=idx_to_links)
    y_link = tf.cast(tf.reduce_any(tf.cast(tf.equal(tgt_idx[...,None],expanded_links),tf.bool),axis=1),dtype=tf.float32)

    return emb1,emb2,y_link

def evaluate_link_pred(emb, indices, links, simsiam, args):
    auc = tf.keras.metrics.AUC()
    f1 = tf.keras.metrics.F1Score(average=None,threshold=0.5)
    
    N = len(links)
    n_batches = math.floor(N / args['batch_size'])
    for b in range(n_batches):
        tmp_emb = emb[b*args['batch_size']:(b+1)*args['batch_size']]
        tmp_indices = indices[b*args['batch_size']:(b+1)*args['batch_size']]
        tmp_links = links[b*args['batch_size']:(b+1)*args['batch_size']]

        emb1,emb2,y_link = prepare_link_data(tmp_emb,tmp_indices,tmp_links,args['batch_size'])
        logging.info(f"y_link shape={y_link.shape},sum={tf.reduce_sum(y_link)}")
        y_link_pred = simsiam.link_predictor.predict((emb1,emb2),batch_size=args['batch_size'])
        y_link = y_link[:,np.newaxis]

        auc.update_state(y_link,y_link_pred)
        f1.update_state(y_link,y_link_pred)
        logging.info(f"batch {b}, auc={auc.result()}, f1={f1.result()}")

    #logging.info(f"Link Pred Evaluation:\n{classification_report(y_link,y_link_pred)}")
    logging.info(f"final AUC = {auc.result()},final F1 = {f1.result()}")

    return f1

########################################################################################

def kmeans_clustering(X,args):
    """
    Use Kmeans to cluster

    X: np.array of N samples * embed dim
    return: labels of shape (N samples,)
    """
    kmeans = MiniBatchKMeans(n_clusters=args['n_cluster'],random_state=args['seed'])
    labels = kmeans.fit_predict(X)
    return labels

def dbscan_clustering(X,args):
    return DBSCAN(metric='cosine').fit_predict(X)



########################################################################################


def evaluate(data, modelpath, args):
    """
    data: {'train':(tr_data,tr_links_data,tr_gt), 'val':(val_data,val_links_data,val_gt), 'test':(te_data,te_links_data,te_gt)}
    """
    labels = data['test'][2]
    data1, data2 = data['test'][0]
    indices,links = data['test'][1]
    
    # load model
    simsiam = SimSiam(args)
    simsiam.built = True
    simsiam.load_weights(modelpath)

    # compute embeddings
    embeddings = compute_embedding(data1,simsiam,args)
    # aug_data_embeddings = compute_embedding(data2,simsiam,args)
    pickle.dump(embeddings,open(args['output_dir']+'/test_data_embeddings.pkl','wb'))
    pickle.dump(labels,open(args['output_dir']+'/test_data_labels.pkl','wb'))
    # plot tsne
    plot_tsne(embeddings,labels,fig_save_path=args['output_dir']+'/org_data_')
    # plot_tsne(aug_data_embeddings,labels,fig_save_path=args['output_dir']+'/aug_data_')

    # cluster
    kmeans_preds = kmeans_clustering(embeddings,args)
    pickle.dump(kmeans_preds,open(args['output_dir']+'/test_data_kmeans_preds.pkl','wb'))
    dbscan_preds = dbscan_clustering(embeddings,args)
    pickle.dump(dbscan_preds,open(args['output_dir']+'/test_data_dbscan_preds.pkl','wb'))

    # evaluate link prediction
    evaluate_link_pred(embeddings, indices, links, simsiam, args)

    return # preds
