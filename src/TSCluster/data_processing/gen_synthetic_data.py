"""
randomly generate n peaks in the timeline
"""

import pandas as pd
import numpy as np
import random
import pickle


def gen_multivar_synthetic_ts(
        length,
        n_dim,
        baseline_poisson_lam=0.05,
        n_peaks=3,
        peak_start_idx=[],
        peak_poisson_lam = None,
        seed=1
):
    """
    generate multivariate synthetic ts
    output: np array of length * n_dim
    """
    # set peak poisson lam
    if peak_poisson_lam is None: peak_poisson_lam = baseline_poisson_lam * 20
    # generate each dimension separately
    ts = np.empty((length,0),dtype=int)
    for d in range(n_dim):
        d_seed = seed + d
        random.seed(d_seed)
        np.random.seed(d_seed)
        ts_tmp = gen_synthetic_ts(
                length,
                baseline_poisson_lam=baseline_poisson_lam,# random.choice(np.arange(baseline_poisson_lam/2,baseline_poisson_lam*2,0.05)),
                n_peaks=n_peaks,#random.choice(np.arange(n_peaks-2,n_peaks+3,1)),
                peak_start_idx=peak_start_idx,
                peak_poisson_lam=peak_poisson_lam,#random.choice(np.arange(peak_poisson_lam/2,peak_poisson_lam*2,0.05)),
                seed=d_seed
        ).reshape((-1,1))
        ts = np.hstack((ts,ts_tmp))

    return ts

def gen_synthetic_ts(
        length,
        baseline_poisson_lam=0.05,
        n_peaks=3,
        peak_start_idx=[],
        peak_poisson_lam = None,
        seed=1
):
    random.seed(seed)
    np.random.seed(seed)
    # set peak poisson lam
    if peak_poisson_lam is None: peak_poisson_lam = baseline_poisson_lam * 20
    # init ts of given length to with a poisson distribution
    ts = np.random.poisson(lam=baseline_poisson_lam,size=length)
    # randomly place n_peaks along the timeline if start indices of peaks not given
    if len(peak_start_idx) != n_peaks:
        peak_start_idx = random.sample(range(length-4),n_peaks) # idx
    # each peak randomly lasts for 1-4 days 
    for i in range(n_peaks):
        peak_length = random.randint(1,4)
        ts_peak = np.random.poisson(lam=peak_poisson_lam,size=peak_length)
        ts[peak_start_idx[i]:peak_start_idx[i]+peak_length] = ts[peak_start_idx[i]:peak_start_idx[i]+peak_length] + ts_peak
    return ts

if __name__ == '__main__':
    """
    generate 3 clusters: 1. all time high posting behavior, 2. normal people with 2 peaks, 3.normal people with 4 different people
    """

    N = 1000
    ts_len = 400
    n_dims = 2

    # TODO: change implementation to let dimensions have different peaks 

    # cluster 1:
    c1 = np.empty((0,ts_len,n_dims))
    for seed in range(N):
        c1 = np.vstack(
            (c1,
            gen_multivar_synthetic_ts(
                ts_len,
                n_dims,
                baseline_poisson_lam=10,
                n_peaks = 2,
                peak_start_idx=[29,318],
                peak_poisson_lam=20,
                seed=seed
            ).reshape((1,ts_len,n_dims)))
        )

    # cluster 2:
    c2 = np.empty((0,ts_len,n_dims))
    for seed in range(N):
        c2 = np.vstack(
            (c2,
            gen_multivar_synthetic_ts(
                ts_len,
                n_dims,
                baseline_poisson_lam=0.05,
                n_peaks = 2,
                peak_start_idx=[29,318],
                peak_poisson_lam=10,
                seed=seed
            ).reshape((1,ts_len,n_dims)))
        )

    # cluster 3:
    c3 = np.empty((0,ts_len,n_dims))
    for seed in range(N):
        c3 = np.vstack(
            (c3,
            gen_multivar_synthetic_ts(
                ts_len,
                n_dims,
                baseline_poisson_lam=0.05,
                n_peaks = 4,
                peak_start_idx=[58,134,168,271],
                peak_poisson_lam=10,
                seed=seed
            ).reshape((1,ts_len,n_dims)))
        )

    syn_data_X = np.concatenate((c1,c2,c3),axis=0).astype(int)
    syn_data_y = np.array([0]*1000+[1]*1000+[2]*1000).astype(int)

    print(syn_data_X.shape)

    # np.savetxt('/nas/home/siyiguo/user_similarity/data/synthetic_data_X.csv',syn_data_X,delimiter=',')
    # np.savetxt('/nas/home/siyiguo/user_similarity/data/synthetic_data_y.csv',syn_data_y,delimiter=',')
    with open('/nas/home/siyiguo/ts_clustering/data/synthetic_data_X.pkl','wb') as f:
        pickle.dump(syn_data_X,f)
    with open('/nas/home/siyiguo/ts_clustering/data/synthetic_data_y.pkl','wb') as f:
        pickle.dump(syn_data_y,f)
    