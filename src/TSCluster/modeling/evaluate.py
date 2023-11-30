import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import pickle

from .model import SimSiam


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
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    plt.xlim((-40, 40))
    plt.ylim((-40, 40))
    fig = sns.scatterplot(x=tsne_results[:, 0],
                          y=tsne_results[:, 1],
                          hue=labels,
                          palette=sns.color_palette(
                              "hls", len(np.unique(labels))),
                          legend="full",
                          alpha=0.3).get_figure()

    fig.savefig(fig_save_path + 'tsne.png', format='png')


def evaluate(data, modelpath, args):
    """
    data: {'train':((tr_data,tr_aug_data),tr_gt), 'val':((val_data,val_aug_data),val_gt), 'test':((te_data,te_aug_data),te_gt)}
    """
    labels = data['test'][1]
    data1, data2 = data['test'][0]
    
    # load model
    simsiam = SimSiam(args)
    simsiam.built = True
    simsiam.load_weights(modelpath)

    # compute embeddings
    embeddings = compute_embedding(data1,simsiam,args)
    aug_data_embeddings = compute_embedding(data2,simsiam,args)
    pickle.dump(embeddings,open(args['output_dir']+'/test_data_embeddings.pkl','wb'))
    pickle.dump(labels,open(args['output_dir']+'/test_data_labels.pkl','wb'))

    # cluster TODO: implement clustering methods

    # plot tsne
    plot_tsne(embeddings,labels,fig_save_path=args['output_dir']+'/org_data_')
    plot_tsne(aug_data_embeddings,labels,fig_save_path=args['output_dir']+'/aug_data_')

