import pickle
import numpy as np
import logging
import tensorflow as tf

from .data_augmentation import data_augmentation

def triplet_formatter(data,max_triplet_len,demo=None):
    """
    Format data from np array to triplets of (timestamp, feature name, value)

    Input:
        data: np.array, N data points * len of ts * n dimension
        max_triplet_len: maximum length of triplets for each data point
        demo: np.array, N data points * n_dim_demo, demographic feature for each data point
    
    Return:
        triplet_data:
        [
            [N datapoints* demo dim], # demographic vector
            [N datapoints* max len of triplets], # timestamps
            [N datapoints* max len of triplets], # values
            [N datapoints* max len of triplets] # feature name (encoded to dummy var)
        ]
        padded with zeros if max len is not met.
        feature dummy var is the index of the dimension + 1 (start at 1)
    """
    N = len(data)

    data_idx, timestamp, feat_dummy = np.where(data!=0)

    timestamp_array = [[] for _ in range(N)]
    values_array = [[] for _ in range(N)]
    feat_dummy_array = [[] for _ in range(N)]

    for i,t,f in zip(data_idx,timestamp,feat_dummy):
        timestamp_array[i].append(t)
        feat_dummy_array[i].append(f+1) # feat dummy start at 1. 0 means not observed
        values_array[i].append(data[i,t,f])

    # cut each ts at max_triplet_len, then pad with zeros - shape of these arrays (N data points * max_triplet_len)
    timestamp_array = np.array([(n+[0]*max_triplet_len)[:max_triplet_len] for n in timestamp_array])
    values_array = np.array([(n+[0]*max_triplet_len)[:max_triplet_len] for n in values_array])
    feat_dummy_array = np.array([(n+[0]*max_triplet_len)[:max_triplet_len] for n in feat_dummy_array])

    if demo is None:
        demo = np.zeros((N,1))

    return [x.astype(float) for x in [demo,timestamp_array,values_array,feat_dummy_array]]


def tr_te_split(ts_data,aug_data,gt=None,tr_frac=0.8,seed=3):
    """
    Args:
        ts_data: list of lists [demo,timestamp_array,values_array,feat_dummy_array], each of these array shape (N * max_triplet_len)
        aug_data: augmented data, same format as ts_data
        gt_data: 1d np array of length N
    """
    # shuffle index and reorder based on the shuffle
    np.random.seed(seed)
    idx = np.arange(len(ts_data[0]))
    np.random.shuffle(idx)
    ts_data = [x[idx] for x in ts_data]
    aug_data = [x[idx] for x in aug_data]
    # partition into tr and te
    tr_data = [x[:int(tr_frac*len(ts_data[0]))] for x in ts_data]
    te_data = [x[int(tr_frac*len(ts_data[0])):] for x in ts_data]
    tr_aug_data = [x[:int(tr_frac*len(ts_data[0]))] for x in aug_data]
    te_aug_data = [x[int(tr_frac*len(ts_data[0])):] for x in aug_data]
    # partition ground truth
    if gt is not None:
        gt = gt[idx]
        tr_gt = gt[:int(tr_frac*len(ts_data[0]))]
        te_gt = gt[int(tr_frac*len(ts_data[0])):]
    else:
        tr_gt, te_gt = None, None

    return (tr_data,tr_aug_data), tr_gt, (te_data,te_aug_data), te_gt


def create_dataset(data_tuple,batch_size):
    """
    create tf.data.Dataset from np arrays
    data_tuple: (ts_data,aug_data)
        ts_data: list of lists [demo,timestamp_array,values_array,feat_dummy_array], each of these array shape (N * max_triplet_len)
    """
    ts_data, aug_data = data_tuple
    ts_data = {
        'demo':ts_data[0],
        'timestamps':ts_data[1],
        'values':ts_data[2],
        'feat':ts_data[3]
    }
    aug_data = {
        'demo':aug_data[0],
        'timestamps':aug_data[1],
        'values':aug_data[2],
        'feat':aug_data[3]
    }

    return tf.data.Dataset.from_tensor_slices((ts_data,aug_data)).batch(batch_size,drop_remainder=True)


def read_data(
        ts_data_dir: str,
        args: dict,
        demo_data_dir: str = None,
        gt_data_dir: str = None,
        max_triplet_len: int = 200,
        augmentation_noisiness: float = 0.3,
        data_split: str = 'no',
        tr_frac: float = 0.8,
        seed: int = 3
):  
    """
    Args:
        ts_data_dir: a pkl file when unpickled is a 3d np array (N data points * ts len * feature dim)
        args: training args, a dict
        demo_data_dir: a pkl file when unpickled is a 2d np array (N data points * demo_dim)
        gt_data_dir: a pkl file when unpickled is a 1d np array (N data points * 1), the gt label of the clusters
        max_triplet_len: maximum length of triplets for each data point
        augmentation_noisiness: how much noise to inject into bootstrapping
        tr_te_split: how to perform split 'no', or 'tr-val' or 'tr-val-te'
        tr_frac: fraction of training data
        seed: random seed

    Return:
        {'train':((tr_data,aug_data),tr_gt), 'val':(val_data,val_gt), 'test':(te_data, te_gt)}
        where tr_data and aug_data are both formatted into triplets
        gt data are 1d np array of length N
    """
    ts_data = pickle.load(open(ts_data_dir,'rb'))
    N = len(ts_data)
    full_ts_range = list(range(ts_data.shape[1]))
    n_feat = ts_data.shape[2]
    demo_dim = 1
    
    demo_data = None
    if demo_data_dir:
        demo_data = pickle.load(open(demo_data_dir,'rb'))
        if len(demo_data) != N:
            raise ValueError('Dimension mismatch between TS and demographic data')
        demo_dim = demo_data.shape[1]

    gt = None
    if gt_data_dir:
        gt = pickle.load(open(gt_data_dir,'rb'))
        if len(gt) != N:
            raise ValueError('Dimension mismatch between TS and groundtruth data')

    # format and combine with demo to input data
    logging.info('start processing data into triplets')
    data = triplet_formatter(ts_data, max_triplet_len, demo_data)
    logging.info(f'finished processing data into triplets, len of data={len(data)}, len of demo data={len(data[0])}, shape of ts data={data[1].shape}')

    # data augmentation
    logging.info('start data augmentation')
    data, aug_data = data_augmentation(data, full_ts_range, n_feat, gen_neg=False, noisiness=augmentation_noisiness, seed=seed)
    logging.info(f'finished data augmentation, len of data={len(aug_data)}, len of demo data={len(aug_data[0])}, shape of ts data={aug_data[1].shape}')

    # train val test split
    if data_split == 'tr-val':
        tr_data, tr_gt, val_data, val_gt = tr_te_split(data,aug_data,gt,tr_frac,seed)
        te_data, te_gt = None, None
        
    elif data_split == 'tr-val-te':
        tr_data, tr_gt, te_data, te_gt = tr_te_split(data,aug_data,gt,tr_frac,seed)
        tr_data, tr_gt, val_data, val_gt = tr_te_split(tr_data[0],tr_data[1],tr_gt,tr_frac,seed)
        
    else:
        tr_data, tr_gt = (data,aug_data), gt
        val_data, val_gt = None, None
        te_data, te_gt = (data,aug_data),gt

    args['n_feat'] = n_feat
    args['demo_dim'] = demo_dim

    return {'train':(tr_data,tr_gt), 'val':(val_data,val_gt), 'test':(te_data, te_gt)}, args