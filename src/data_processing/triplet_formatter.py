import numpy as np

def triplet_formatter(data,max_len,demo=None):
    """
    Format data from np array to triplets of (timestamp, feature name, value)

    Input:
        data: np.array, N data points * len of ts * n dimension
        max_len: maximum length of triplets for each data point
        demo: np.array, N data points * n_dim_demo, demographic feature for each data point
    Return:
        triplet_data:
        [
            [N datapoints* max len of triplets], # timestamps
            [N datapoints* max len of triplets], # values
            [N datapoints* max len of triplets] # feature name (encoded to dummy var)
        ]
        padded with zeros if max len is not met.
        feature dummy var is the index of the dimension + 1 (start at 1)
    """
    N = len(data)

    data_idx, timestamp, feat_dummy = np.where(data!=0)

    timestamp_array = [[]]*N
    values_array = [[]]*N
    feat_dummy_array = [[]]*N

    for i,t,f in zip(data_idx,timestamp,feat_dummy):
        timestamp_array[i].append(t)
        feat_dummy_array[i].append(f)
        values_array[i].append(data[i,t,f])

    timestamp_array = np.array(timestamp_array)
    values_array = np.array(values_array)
    feat_dummy_array = np.array(feat_dummy_array)

    if demo is None:
        demo = np.zeros((N,2))

    return [demo,timestamp_array,values_array,feat_dummy_array]