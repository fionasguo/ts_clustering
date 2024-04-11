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


def tr_te_split(ts_data,aug_data,gt=None,indices=None,links_data=None,tr_frac=0.8,seed=3):
    """
    Args:
        ts_data: list of lists [demo,timestamp_array,values_array,feat_dummy_array], each of these array shape (N * max_triplet_len)
        aug_data: augmented data, same format as ts_data
        gt_data: 1d np array of length N
        links_data: 2d array (N * max # links across users)
    """
    # shuffle index and reorder based on the shuffle
    np.random.seed(seed)
    idx = np.arange(len(ts_data[0]))
    np.random.shuffle(idx)
    ts_data = [x[idx] for x in ts_data]
    if aug_data is not None:
        aug_data = [x[idx] for x in aug_data]
    # partition into tr and te
    tr_data = [x[:int(tr_frac*len(ts_data[0]))] for x in ts_data]
    te_data = [x[int(tr_frac*len(ts_data[0])):] for x in ts_data]
    if aug_data is not None:
        tr_aug_data = [x[:int(tr_frac*len(ts_data[0]))] for x in aug_data]
        te_aug_data = [x[int(tr_frac*len(ts_data[0])):] for x in aug_data]
    else:
        tr_aug_data,te_aug_data = None, None

    # partition ground truth
    if gt is not None:
        gt = gt[idx]
        tr_gt = gt[:int(tr_frac*len(ts_data[0]))]
        te_gt = gt[int(tr_frac*len(ts_data[0])):]
    else:
        tr_gt, te_gt = None, None

    # link prediction data
    if links_data is not None:
        indices = indices[idx]
        links_data = links_data[idx]
        tr_links_data = links_data[:int(tr_frac*len(ts_data[0]))]
        te_links_data = links_data[int(tr_frac*len(ts_data[0])):]
        tr_indices = indices[:int(tr_frac*len(ts_data[0]))]
        te_indices = indices[int(tr_frac*len(ts_data[0])):]
    else:
        tr_links_data, te_links_data, tr_indices, te_indices = None, None, None, None

    return (tr_data,tr_aug_data), (tr_indices,tr_links_data), tr_gt, (te_data,te_aug_data), (te_indices,te_links_data), te_gt

def rvw_blah(ts_data,aug_data,gt=None,indices=None,links_data=None,tr_frac=0.8,seed=3):
    tr_idx = [ 55026,  32302, 116711,  55009, 115248,  82134,  54991,  12019,
        41424,  55021,  39729,  54999,  55033,  55011,  30263,  64876,
        55020,  88313,  51210,  51238,  51181,   7269,  55007,  51193,
        51217,  36660,  51220,  10790,  51206,  55030,  51228,  51234,
        54203,  51219,  13411,  51201,  51188,  51229,   5332,  26048,
        39047,  51180,  51235,  51237,  51189,  51197,  51195, 118737,
        51192,  54825,  51190,  51208,  51207,  54998,  52439,  13210,
        51231,  51973,  51225,  51203,  48845,  51240,  51179,  51194,
        51178,  51185,  51216, 111133, 115263,  52853,  29920,  51212,
        51187,  51236,  81796,  51182,  51204,  69912,  51202,  51199,
        51213, 108006,  79251,  51215,  51183,  84107,  54997,  51243,
        98906,  51200,  61331,  12132,   6934,  21464,  56890,  55022,
        35717,  55006,  18859,  55002,  22376,  26817,  23349,  84918,
        61347, 114975,  23347,  57591,  23344,  23354,  23339,  10490,
        55035, 103708, 119338,  64316,  43773,   9587, 106035, 115991,
        94969,  62829,   9878,   9871,  21255,  44818, 103701,  44897,
       103514,   1427, 106431,  45727,  14796, 115879,   9371, 115815,
       102786, 116008,  44105,  44860, 116139,   1428,  72893, 109644,
        93107, 109653, 103049,  78040,  94410,  75348,  74636,  88691,
        95719,  99363, 104109,  76906,  98773,  90845,  98719,  90638,
        90810,  78272,  72814,  68723, 110051,  72861, 103139, 105059,
        68816,  77918, 101801,  91019, 111143, 109899,  67113, 114134,
       110303,  69168, 102629, 105076,  91576, 112741,  71599,  83828,
        94434, 119991,  71870,  80926,  69820,  83367,  72826, 100824,
        69319, 108380,  70648,  90803,  67504,  97584,  98311, 108376,
        68707, 115960,  75327,  89832,  85171,  92417,  98081, 106033,
       110916,  90330, 100692,  86526,  87129,  92924, 105021, 113635,
        74851,  99872,  90464,  99560, 101634, 113459,  90596, 120530,
       119919, 108747, 117478,  14674,  43682,  48669,  58543,  30299,
       103414,  60631,  57552,   6885,  61330,   9726,  30292,   9552,
        32300,   6564,  12357,  74125, 106045,   9060,  23342,  82984,
        36406,  60166,  85518,   3492,   5351,  45818, 101214,  47306,
        71161,  49341,  44592,  98871,  34819,  44816,  25065, 111216,
        84479,  45368, 100810,   2261,  48667,  31187, 105963,   2315,
         9269, 113314, 109619,  59426,  59448,  89732,    947,  74815,
         3264,  67905,  19752,  90829,  15830,  98305,  23623,  97119,
        77524,  47045,   9163,  97025,  99375,  96038,  99444,  47320,
        81162,  91770,  83135,  90922,  98689,  14846,  20419, 111054,
        67365,  14536,  93199,  75190,  45121, 112762,  55210,   2614,
         2130,  91767,  87081,  38577,  45508,  26127,   7329,  44483,
       109595, 102609,  23333,   5552,  58561,  81574, 100094,  49867,
       116114,  68601,  29035, 101995,   5876,  94462, 112684,  41888,
        15728,   8179,  42253,   8798,  55481,  19213,  23410,  50556,
        87418,  60169, 101519,  71337,   3881,  69417,  30572,  12695,
        66166,  16732,  84761,  68031,  64506, 109039]
    
    # idx for autonomy health issue
    # tr_idx = [504, 505, 812, 864, 882, 1003, 1258, 1358, 1527, 2088, 2176, 2291, 2550, 2674, 2694, 2831, 3151, 3359, 3448, 3478, 3515, 3552, 3621, 3640, 3753, 4078, 4555, 4586, 4984, 5456, 5539, 5843, 7962, 8036, 8359, 8692, 8696, 8698, 8700, 8702, 8806, 9372, 9757, 10032, 10819, 11284, 11286, 12005, 13016, 13395, 13659, 14525, 14980, 15773, 15892, 16517, 16669, 16816, 16856, 16951, 16952, 16964, 16974, 17055, 17199, 17326, 17834, 17936, 18437, 18439, 18522, 18706, 18893, 19372, 19373, 19374, 19375, 19376, 19377, 19378, 19379, 19381, 19382, 19383, 19384, 19386, 19387, 19388, 19390, 19391, 19392, 19393, 19394, 19395, 19397, 19398, 19399, 19400, 19404, 19405, 19407, 19410, 19411, 19412, 19413, 19415, 19988, 20487, 20701, 20763, 20768, 20769, 20770, 20773, 20774, 20781, 20782, 20785, 20788, 20791, 20792, 21515, 21759, 22451, 22461, 22716, 22717, 22872, 23108, 23109, 23115, 23604, 24170, 24385, 25299, 25538, 25858, 26316, 26347, 26828, 26909, 26999, 27451, 28246, 28400, 28453, 28463, 29104, 29330, 29459, 29594, 30017, 30702, 30816, 31086, 31583, 31733, 32030, 32181, 32299, 32352, 32446, 32580, 33195, 33345, 33702, 33839, 34226, 34454, 34499, 34619, 34626, 34654, 34675, 34967, 36105, 36126, 36317, 36766, 37699, 37878, 37917, 37930, 38102, 38694, 38861, 38973, 39465, 39577, 39742, 39784, 39869, 40749, 40776, 40929, 41530, 41935, 42143, 42150, 42165, 42168, 42264, 42732, 42793, 43395, 43412, 43422, 43644, 44642, 44662, 44693, 44704, 44708, 44750, 44762, 44969, 45742, 45980]

    # # idx for religion fetal rights issue
    # tr_idx = [450, 606, 1572, 2532, 3299, 3979, 4957, 5229, 5396, 5996, 6248, 7240, 7772, 7835, 8121, 8346, 8457, 8563, 8673, 8777, 8778, 8779, 8780, 8985, 10248, 10671, 11549, 11742, 11764, 11879, 11994, 12248, 12571, 12630, 13093, 13698, 13978, 14305, 14374, 14581, 15053, 15237, 15843, 15926, 16083, 16513, 16653, 16709, 16753, 16936, 16979, 17004, 17159, 17277, 17332, 17728, 17739, 17805, 18228, 18280, 18389, 18882, 19422, 19431, 19441, 19447, 19469, 19999, 20085, 20186]

    # partition into tr and te
    tr_data = [x[tr_idx] for x in ts_data]
    te_data = [np.delete(x,tr_idx,axis=0) for x in ts_data]
    tr_aug_data = None #[x[tr_idx] for x in aug_data]
    te_aug_data = None #[np.delete(x,tr_idx,axis=0) for x in aug_data]
    # partition ground truth
    if gt is not None:
        tr_gt = gt[tr_idx]
        te_gt = np.delete(gt,tr_idx,axis=0)
    else:
        tr_gt, te_gt = None, None
    # link prediction data
    if links_data is not None:
        tr_indices = indices[tr_idx]
        tr_links_data = links_data[tr_idx]
        te_links_data = np.delete(links_data,tr_idx,axis=0)
        te_indices = np.delete(indices,tr_idx,axis=0)
    else:
        tr_links_data, te_links_data, tr_indices, te_indices = None, None, None, None
    # val and te
    (te_data,te_aug_data),(te_indices,te_links_data), te_gt, (val_data,val_aug_data),(val_indices,val_links_data), val_gt = tr_te_split(te_data,te_aug_data,gt=te_gt,indices=te_indices,links_data=te_links_data,tr_frac=0.5,seed=seed)

    return (tr_data,tr_aug_data), (tr_indices,tr_links_data), tr_gt, (val_data,val_aug_data), (val_indices,val_links_data), val_gt, (te_data,te_aug_data), (te_indices,te_links_data), te_gt


def create_dataset(data,batch_size):
    """
    create tf.data.Dataset from np arrays
    data: ((ts_data,aug_data),(indices,links))
        ts_data: list of lists [demo,timestamp_array,values_array,feat_dummy_array], each of these array shape (N * max_triplet_len)
    """
    ((ts_data,aug_data),(indices,links)) = data
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

    return tf.data.Dataset.from_tensor_slices((ts_data,aug_data,indices,links)).batch(batch_size,drop_remainder=True)

def binarize_gt(gt):
    biggest_cluster = np.max(gt)
    mask = gt < biggest_cluster

    binarized_gt = np.zeros(gt.shape)
    binarized_gt[mask] = 1
    return binarized_gt


def read_data(
        ts_data_dir: str,
        args: dict,
        demo_data_dir: str = None,
        gt_data_dir: str = None,
        links_data_dir: str = None,
        max_triplet_len: int = 200,
        data_aug: bool = True,
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
        links_data_dir: retweet/following links prediction data, each row is the indices of other users that this user is linked to, 
            the number of columns equal to the max number of links across all users. padded with -1.
            a pkl file of a 2d np array (N data points * max # of links)
        max_triplet_len: maximum length of triplets for each data point
        data_aug: whether to perform augmentation
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

    if len(ts_data.shape) == 2:
        # add a dim
        ts_data = ts_data[:, :, np.newaxis]

    N = len(ts_data)

    full_ts_range = list(range(ts_data.shape[1]))

    n_feat = ts_data.shape[2]
    logging.info(f"n_feat={n_feat}")
    demo_dim = 1
    
    demo_data = None
    if demo_data_dir:
        demo_data = pickle.load(open(demo_data_dir,'rb'))
        if len(demo_data) != N:
            raise ValueError('Dimension mismatch between TS and demographic data')
        demo_dim = demo_data.shape[1]
        logging.info(f"check demo data - demo_dim={demo_dim}, #nans={np.isnan(demo_data).sum()}")

    gt = None
    if gt_data_dir:
        gt = pickle.load(open(gt_data_dir,'rb'))
        gt = gt.astype(float)
        if len(gt) != N:
            raise ValueError('Dimension mismatch between TS and groundtruth data')
        if args['binarize_gt']:
            gt = binarize_gt(gt)
            args['n_classes'] = 2
        else:
            args['n_classes'] = len(np.unique(gt))
        logging.info(f"check gt data - n_classes={args['n_classes']},#nans={np.isnan(gt).sum()}")

    links_data = None
    if links_data_dir:
        links_data = pickle.load(open(links_data_dir,'rb'))
        links_data = links_data.astype(int)
        logging.info(f"link prediction data available, shape={links_data.shape}")

    # format and combine with demo to input data
    logging.info('start processing data into triplets')
    data = triplet_formatter(ts_data, max_triplet_len, demo_data)
    logging.info(f'finished processing data into triplets, len of data={len(data)}, len of demo data={len(data[0])}, shape of ts data={data[1].shape}')

    # data augmentation
    aug_data = None
    if data_aug:
        logging.info('start data augmentation')
        data, aug_data = data_augmentation(data, full_ts_range, n_feat, noisiness=augmentation_noisiness, seed=seed)
        logging.info(f'finished data augmentation, len of data={len(aug_data)}, len of demo data={len(aug_data[0])}, shape of ts data={aug_data[1].shape}')

    # train val test split
    if data_split == 'tr-val':
        tr_data, tr_links_data, tr_gt, val_data, val_links_data, val_gt = tr_te_split(data,aug_data,gt,np.arange(len(data[0])),links_data,tr_frac,seed)
        te_data, te_links_data, te_gt = None, None
        logging.info(f"tr_data shape={tr_data[0][1].shape}, tr_gt shape={tr_gt.shape}, #label '1'={np.sum(tr_gt)}")
        logging.info(f"val_data shape={val_data[0][1].shape}, val_gt shape={val_gt.shape}, #label '1'={np.sum(val_gt)}")
        if links_data is not None: logging.info(f"tr indices shape={tr_links_data[0].shape}, tr_links shape={tr_links_data[1].shape}")
        
    elif data_split == 'tr-val-te':
        # tr_data, tr_links_data, tr_gt, te_data, te_links_data, te_gt = tr_te_split(data,aug_data,gt,np.arange(len(data[0])),links_data,tr_frac,seed)
        # te_data, te_links_data, te_gt, val_data, val_links_data, val_gt = tr_te_split(te_data[0],te_data[1],te_gt,te_links_data[0],te_links_data[1],0.5,seed)
        tr_data, tr_links_data, tr_gt, val_data, val_links_data, val_gt, te_data, te_links_data, te_gt = rvw_blah(data,aug_data,gt,np.arange(len(data[0])),links_data,tr_frac,seed)
        logging.info(f"tr_data shape={tr_data[0][1].shape}, tr_gt shape={tr_gt.shape}, #label '1'={np.sum(tr_gt)}")
        logging.info(f"val_data shape={val_data[0][1].shape}, val_gt shape={val_gt.shape}, #label '1'={np.sum(val_gt)}")
        logging.info(f"te_data shape={te_data[0][1].shape}, te_gt shape={te_gt.shape}, #label '1'={np.sum(te_gt)}")
        if links_data is not None: logging.info(f"tr indices shape={tr_links_data[0].shape}, tr_links shape={tr_links_data[1].shape}")
        
    else:
        tr_data, tr_links_data, tr_gt = (data,aug_data), (np.arange(len(data[0])),links_data), gt
        val_data, val_links_data, val_gt = None, None, None
        te_data, te_links_data, te_gt = (data,aug_data), (np.arange(len(data[0])),links_data), gt
        logging.info(f"tr_data shape={tr_data[0][1].shape}, tr_gt shape={tr_gt.shape}, #label '1'={np.sum(tr_gt)}")
        logging.info(f"te_data shape={te_data[0][1].shape}, te_gt shape={te_gt.shape}, #label '1'={np.sum(te_gt)}")
        if links_data is not None: logging.info(f"tr indices shape={tr_links_data[0].shape}, tr_links shape={tr_links_data[1].shape}")
        
    args['n_feat'] = n_feat
    args['demo_dim'] = demo_dim

    return {'train':(tr_data,tr_links_data,tr_gt), 'val':(val_data,val_links_data,val_gt), 'test':(te_data,te_links_data,te_gt)}, args