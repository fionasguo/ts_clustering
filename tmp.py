import os,sys
import pickle
import numpy as np
from sklearn.manifold import TSNE


data_dir = '/nas/eclairnas01/users/siyiguo/ts_clustering/'
folder = sys.argv[1]

feats = pickle.load(open(data_dir+folder+'/test_data_embeddings.pkl','rb'))
tsne = TSNE(n_components=2, verbose=0)
tsne_results = tsne.fit_transform(feats)

pickle.dump(tsne_results,open(data_dir+folder+'/test_data_embeddings_tsne.pkl','wb'))
