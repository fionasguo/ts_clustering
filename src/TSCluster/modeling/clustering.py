from sklearn.cluster import MiniBatchKMeans


def kmeans_clustering(X,args):
    """
    Use Kmeans to cluster

    X: np.array of N samples * embed dim
    return: labels of shape (N samples,)
    """
    kmeans = MiniBatchKMeans(n_clusters=args['n_cluster'],random_state=args['seed'])
    labels = kmeans.fit_predict(X)
    return labels
