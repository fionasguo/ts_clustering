import random
import numpy as np
import math


def shuffle_segments(timestamps, feat_dummies, values, full_ts_range, k=4):
    """
    randomly select k time points that have 0 or baseline value
    shuffle these k+1 blocks by shifting timestamps
    """
    zero_valued_timestamps = [t for t in full_ts_range if t not in timestamps]
    # if len(zero_valued_timestamps) < k:
        # if there's nonbaseline values also works

    return


## generate positive examples by bootstrapping
def data_augmentation(ts_data, full_ts_range, n_feat, noisiness=0.3, shuffle_segments=False, seed=3):
    """
    Generate one positive example using bootstrapping

    Args:
        ts_data: triplet data [demo,timestamp_array,values_array,feat_dummy_array]
                demo shape: N * demo_dim
                other array shape: N * max_triplet_len
        full_ts_range: a list, all possible timestamps
        n_feat: number of features. feature dummies start from 1.
        noisiness: how much noise to inject into bootstrapping
        shuffle_segments: whether to randomly shuffle segments after bootstrapping
            for cases eg coord campaign, some users only have very high volumn of tweets in a short period of time and zero everywhere else
        seed: random seed

    Return:
        aug_data: same format as ts_data
    """
    N = len(ts_data[0]) # number of data points
    max_triplet_len = ts_data[1].shape[1]
    
    aug_timestamp_array = [[] for _ in range(N)]
    aug_values_array = [[] for _ in range(N)]
    aug_feat_dummy_array = [[] for _ in range(N)]

    for n in range(N):
        for f in range(1,n_feat+1):
            # for each data point for each feat, do bootstrapping separately.
            triplet_len = np.sum(ts_data[3][n]==f)
            if triplet_len <= 5:
                # if number of observations in one feature dim too small, consider these as noise
                # an augmentation would be to construct another ts with 
                # randomly distributed noise (1 - 5 signals) in the full ts range
                random.seed(random.choice(list(range(100000))))
                random_time_points = random.choices(full_ts_range,k=random.choice(list(range(0,6,1))))
                for i in random_time_points:
                    aug_timestamp_array[n].append(i)
                    aug_feat_dummy_array[n].append(f) # feat dummy start at 1. 0 means not observed
                    aug_values_array[n].append(1)
            else:
                # bootstrapping
                tmp_timestamps = []
                tmp_feat_dummy = []
                tmp_values = []
                random.seed(seed+n*n_feat+f)
                # a range of the number of data points as the output of bootstrapping, 
                # e.g. original data have 50 data points, we can bootstrap 45 data points, or 55.
                btstrp_count_range = list(range(int(triplet_len*(1-noisiness)), int(triplet_len*(1+noisiness))+2, 1))
                # a range of scale factors for the signals
                btstrp_scale_range = list(np.arange((1-noisiness),(1+noisiness),0.05))
                # a range of lag time
                btstrp_lag_range = [math.ceil(x) for x in list(np.arange(-(noisiness*5),(noisiness*5)+0.5,0.5))]
                # by bootstrapping, generate indices where there will be data
                btstrpd_triplet_idx = random.choices(list(range(triplet_len)),k=random.choice(btstrp_count_range))
                for i in btstrpd_triplet_idx:
                    random.seed(seed+n*n_feat+len(btstrpd_triplet_idx)*f+i)
                    btstrp_scale = random.choice(btstrp_scale_range)
                    btstrp_lag = random.choice(btstrp_lag_range)
                    aug_idx = i + btstrp_lag
                    if aug_idx >= len(ts_data[1][n][ts_data[3][n]==f]):
                        aug_idx = len(ts_data[1][n][ts_data[3][n]==f])-1
                    elif aug_idx < 0:
                        aug_idx = 0

                    tmp_timestamps.append(ts_data[1][n][ts_data[3][n]==f][aug_idx])
                    tmp_feat_dummy.append(f)
                    tmp_values.append(btstrp_scale*(ts_data[2][n][ts_data[3][n]==f][aug_idx]))
                
                # shuffle segments
                # if shuffle_segments:
                #     tmp_timestamps, tmp_feat_dummy, tmp_values = shuffle_segments(tmp_timestamps,tmp_feat_dummy,tmp_values,full_ts_range)
                
                aug_timestamp_array[n].extend(tmp_timestamps)
                aug_feat_dummy_array[n].extend(tmp_feat_dummy)
                aug_values_array[n].extend(tmp_values)
    
    # cut each ts at max_triplet_len, then pad with zeros - shape of these arrays (N data points * max_triplet_len)
    aug_timestamp_array = np.array([(n+[0]*max_triplet_len)[:max_triplet_len] for n in aug_timestamp_array])
    aug_values_array = np.array([(n+[0]*max_triplet_len)[:max_triplet_len] for n in aug_values_array])
    aug_feat_dummy_array = np.array([(n+[0]*max_triplet_len)[:max_triplet_len] for n in aug_feat_dummy_array])
    
    aug_data = [x.astype(float) for x in [ts_data[0], aug_timestamp_array,aug_values_array,aug_feat_dummy_array]]

    return ts_data, aug_data
