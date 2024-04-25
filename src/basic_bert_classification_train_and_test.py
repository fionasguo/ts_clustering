import os
import time
import logging
import pickle
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import losses, Model
from tensorflow.keras.layers import Dense, ReLU, Dropout, Input
from tensorflow.keras.utils import to_categorical

from TSCluster import set_seed, get_training_args, create_logger
from TSCluster import read_data
from TSCluster import SimSiam, Trainer
from TSCluster import plot_tsne



def build_classifier(args,input_dim=768):
    n_classes = 1 if args['n_classes']==2 else args['n_classes']

    input_x = Input((input_dim,))
    x = Dense(args['embed_dim'],activation='relu')(input_x)
    x = Dropout(args['dropout'])(x)
    x = Dense(3,activation='softmax')(x)

    classifier = Model(input_x, x, name='classifier')

    return classifier

def train_classifier(train_data,val_data,args,simsiam=None,savepath=None):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=args['patience'], restore_best_weights=True
            )
    ]
    
    classifier = build_classifier(args)
    
    loss_fn = losses.CategoricalCrossentropy(from_logits=False)

    classifier.compile(
        loss=loss_fn,
        optimizer='adam',
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.F1Score(average=None,threshold=0.5)
            ]
    )
    
    history = classifier.fit(
        train_data,
        epochs=args['epoch'],
        validation_data=val_data,
        callbacks=callbacks
    )

    if savepath is None: savepath = args['output_dir']+'/classifier_model_weights.h5'
    classifier.save_weights(savepath)
    
    logging.info('training finished. log:')
    logging.info(history.history)

    # plot tsne
    logging.info("plot tsne for trianing and val data")
    classifier.summary(print_fn=logging.info)

    return classifier

def test_classifier(test_data, args, classifier):
    logging.info("Testing Classifier")
    if classifier is None: 
        if args.get('trained_classifier_model_dir') is not None:
            eval_model_path = args['trained_classifier_model_dir']
            classifier = build_classifier(args,simsiam=None)
            classifier.built = True
            classifier.load_weights(eval_model_path)

            loss_fn = losses.CategoricalCrossentropy(from_logits=False)

            classifier.compile(
                loss=loss_fn,
                optimizer='adam',
                metrics=[
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.F1Score(average=None,threshold=0.5)
                    ]
            )
            logging.info('Classifier loaded')
            classifier.summary(print_fn=logging.info)
        else:
            raise ValueError('Please provide a model for classification evaluation.')

    eval_results = classifier.evaluate(test_data,batch_size=args['batch_size'])
    test_preds_conf = classifier.predict(test_data,batch_size=args['batch_size'])
    pickle.dump(test_preds_conf,open(args['output_dir']+'/classifier_test_preds_conf.pkl','wb'))


    logging.info(f"Finished evaluating test data.\nLoss: {eval_results[0]}, Precision:{eval_results[1]}, Recall:{eval_results[2]}, AUC-ROC:{eval_results[3]}, F1 score:{eval_results[4]}")
    logging.info(eval_results)
    
    test_preds = np.argmax(test_preds_conf,axis=1)
    test_labels = np.asarray(list(test_data.map(lambda X,y:y)))
    print(test_preds.shape,test_labels.shape)
    if test_labels.shape[-1] == 1:
        test_labels = test_labels.flatten()
        print('1',test_labels.shape)
    else:
        test_labels = test_labels.reshape((-1,test_labels.shape[-1]))
        test_labels = np.argmax(test_labels,axis=1).flatten()
        print('2',test_labels.shape)
    logging.info(classification_report(test_labels,test_preds))

    return eval_results


def tr_te_split(ts_data,gt,tr_frac=0.8,seed=3):
    # shuffle index and reorder based on the shuffle
    np.random.seed(seed)
    idx = np.arange(len(ts_data))
    np.random.shuffle(idx)

    ts_data = ts_data[idx,:]
    gt = gt[idx]
    # partition into tr and te
    tr_data = ts_data[:int(tr_frac*len(ts_data))]
    te_data = ts_data[int(tr_frac*len(ts_data)):]
    # partition ground truth
    tr_gt = gt[:int(tr_frac*len(ts_data))]
    te_gt = gt[int(tr_frac*len(ts_data)):]

    return tr_data, tr_gt, te_data, te_gt


def read_data(
        ts_data_dir: str,
        args: dict,
        gt_data_dir: str,
        data_split: str = 'no',
        tr_frac: float = 0.8,
        seed: int = 3
):  
    ts_data = pickle.load(open(ts_data_dir,'rb'))
    ts_data = ts_data.reshape((ts_data.shape[0],ts_data.shape[2]))

    N = len(ts_data)

    n_feat = ts_data.shape[1]
    logging.info(f"n_feat={n_feat}")

    gt = pickle.load(open(gt_data_dir,'rb'))
    gt = gt.astype(float)

    if len(gt) != N:
        raise ValueError('Dimension mismatch between TS and groundtruth data')
    args['n_classes'] = len(np.unique(gt))
    logging.info(f"check gt data - n_classes={args['n_classes']},#nans={np.isnan(gt).sum()}")
    # one-hot
    encoder = LabelEncoder()
    encoder.fit(gt)
    gt = encoder.transform(gt)
    # convert integers to dummy variables (i.e. one hot encoded)
    gt = to_categorical(gt)

    # train val test split
    if data_split == 'tr-val-te':
        tr_data, tr_gt, te_data, te_gt = tr_te_split(ts_data,gt,tr_frac,seed)
        te_data, te_gt, val_data, val_gt = tr_te_split(te_data,te_gt,0.5,seed)
        logging.info(f"tr_data shape={tr_data.shape}, tr_gt shape={tr_gt.shape}, #label '1'={np.sum(tr_gt)}")
        logging.info(f"val_data shape={val_data.shape}, val_gt shape={val_gt.shape}, #label '1'={np.sum(val_gt)}")
        logging.info(f"te_data shape={te_data.shape}, te_gt shape={te_gt.shape}, #label '1'={np.sum(te_gt)}")
        
    else:
        tr_data, tr_gt = ts_data, gt
        val_data, val_gt = None, None
        te_data, te_gt = ts_data, gt
        logging.info(f"tr_data shape={tr_data.shape}, tr_gt shape={tr_gt.shape}, #label '1'={np.sum(tr_gt)}")
        logging.info(f"te_data shape={te_data.shape}, te_gt shape={te_gt.shape}, #label '1'={np.sum(te_gt)}")
        
    args['n_feat'] = n_feat

    return {
        'train': tf.data.Dataset.from_tensor_slices((tr_data, tr_gt)).batch(args['batch_size'],drop_remainder=True), 
        'val': tf.data.Dataset.from_tensor_slices((val_data, val_gt)).batch(args['batch_size'],drop_remainder=True), 
        'test': tf.data.Dataset.from_tensor_slices((te_data, te_gt)).batch(args['batch_size'],drop_remainder=True)
        }, args

if __name__ == '__main__':
    # args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    mode, args = get_training_args(root_dir)

    # logger
    create_logger(args)

    logging.info('Configurations:')
    logging.info(args)

    # set seed
    set_seed(args['seed'])

    if args['gt_dir'] is None:
        raise ValueError('Ground truth data not provided')

    # train / test
    classifier = None
    if mode == 'train':
        # datasets: {'train':(tr_data,tr_gt), 'val':(val_data,val_gt), 'test':(te_data, te_gt)}
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    gt_data_dir=args['gt_dir'],
                    data_split='tr-val-te',
                    tr_frac=args['tr_frac'],
                    seed=args['seed']
                )
        classifier = train_classifier(datasets['train'],datasets['val'],args)

    elif mode == 'test':
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    gt_data_dir=args['gt_dir'],
                    data_split='no',
                    tr_frac=args['tr_frac'],
                    seed=args['seed']
                )
        test_classifier(datasets['test'],args,classifier)

    elif mode == 'train_test':
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    gt_data_dir=args['gt_dir'],
                    data_split='tr-val-te',
                    tr_frac=args['tr_frac'],
                    seed=args['seed']
                )

        classifier = train_classifier(datasets['train'],datasets['val'],args)

        test_classifier(datasets['test'],args,classifier)
    else:
        ValueError('Please specify command arg -m or --mode to be train,test,or train_test')
