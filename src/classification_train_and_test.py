import os
import time
import logging
import logging
import numpy as np

import tensorflow as tf
from tensorflow.keras import losses, Model
from tensorflow.keras.layers import Dense, ReLU, Dropout

from TSCluster import set_seed, get_training_args, create_logger
from TSCluster import read_data
from TSCluster import SimSiam, Trainer
from TSCluster import plot_tsne


def create_classification_dataset(data_tuple,batch_size):
    """
    create tf.data.Dataset from np arrays
    data_tuple: ((ts_data,aug_data), y)
        ts_data: list of lists [demo,timestamp_array,values_array,feat_dummy_array], each of these array shape (N * max_triplet_len)
    """
    X, y = data_tuple
    ts_data = X[0]
    ts_data = {
        'demo':ts_data[0],
        'timestamps':ts_data[1],
        'values':ts_data[2],
        'feat':ts_data[3]
    }

    return tf.data.Dataset.from_tensor_slices((ts_data,y)).batch(batch_size,drop_remainder=True)


def train_simsiam(data, args):
    start_time = time.time()
    logging.info('Start training SimSiam...')

    # set up trainer
    trainer = Trainer(data, args)

    trainer.train()

    logging.info(
        f"Finished training SimSiam. Time: {time.time()-start_time}"
    )

    return trainer


def build_classifier(args,simsiam=None):
    if simsiam is None:
        simsiam = SimSiam(args)
        simsiam.built = True
        if args['trained_model_dir'] is not None:
            simsiam.load_weights(args['trained_model_dir'])
            logging.info('loading simsiam model, weights are loaded')
        else:
            logging.info('loading simsiam model, weights are not loaded, using random weights')
    
    n_classes = 1 if args['n_classes']==2 else args['n_classes']

    # x = simsiam.predictor(simsiam.encoder.output)
    x = Dense(args['embed_dim'],activation='relu')(simsiam.encoder.output)
    x = Dropout(args['dropout'])(x)
    x = Dense(n_classes,activation='sigmoid')(x)

    classifier = Model(simsiam.encoder.input, x, name='classifier')

    return classifier

def train_classifier(train_data,val_data,args,simsiam=None,savepath=None):
    # data
    train_dataset = create_classification_dataset(train_data, args['batch_size'])
    val_dataset = create_classification_dataset(val_data, args['batch_size'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=args['patience'], restore_best_weights=True
            )
    ]
    
    classifier = build_classifier(args,simsiam)
    
    if args['n_classes'] == 2:
        loss_fn = losses.BinaryCrossentropy(from_logits=False)
    else:
        loss_fn = losses.CategoricalCrossentropy(from_logits=False)

    classifier.compile(loss=loss_fn,
              optimizer='adam',
              metrics=tf.keras.metrics.F1Score(average=None,threshold=0.5))
    
    history = classifier.fit(
        train_dataset,
        epochs=args['epoch'],
        validation_data=val_dataset,
        callbacks=callbacks
    )

    if savepath is None: savepath = args['output_dir']+'/classifier_model_weights.h5'
    classifier.save_weights(savepath)
    
    logging.info('training finished. log:')
    logging.info(history.history)

    # plot tsne
    logging.info("plot tsne for trianing and val data")
    classifier.summary(print_fn=logging.info)
    embedding_model = Model(inputs=classifier.input, outputs=classifier.get_layer('embed_output').output)
    tr_emb = embedding_model.predict(train_dataset.map(lambda X, y: X))
    tr_labels = np.asarray(list(train_dataset.map(lambda X,y:y))).flatten()
    val_emb = embedding_model.predict(val_dataset.map(lambda X, y: X))
    val_labels = np.asarray(list(val_dataset.map(lambda X,y:y))).flatten()

    plot_tsne(tr_emb, tr_labels, fig_save_path=args['output_dir']+'/classifier_train_data_')
    plot_tsne(val_emb, val_labels, fig_save_path=args['output_dir']+'/classifier_val_data_')

    return classifier

def test_classifier(test_data, args, classifier):
    if classifier is None: 
        if args.get('trained_classifier_model_dir') is not None:
            eval_model_path = args['trained_classifier_model_dir']
            classifier = build_classifier(args,simsiam=None)
            classifier.built = True
            classifier.load_weights(eval_model_path)
        else:
            raise ValueError('Please provide a model for classification evaluation.')

    test_dataset = create_classification_dataset(test_data, args['batch_size'])

    eval_results = classifier.evaluate(test_dataset,batch_size=args['batch_size'])

    # plot tsne
    embedding_model = Model(inputs=classifier.input, outputs=classifier.get_layer('embed_output').output)
    test_emb = embedding_model.predict(test_dataset.map(lambda X, y: X))
    test_labels = np.asarray(list(test_dataset.map(lambda X,y:y))).flatten()
    plot_tsne(test_emb, test_labels, fig_save_path=args['output_dir']+'/classifier_test_data_')

    logging.info(f"Finished evaluating test data.\nLoss: {eval_results[0]}, F1 score:{eval_results[1]}")

    return eval_results


if __name__ == '__main__':
    # logger
    create_logger()

    # args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    mode, args = get_training_args(root_dir)

    # set seed
    set_seed(args['seed'])

    if args['gt_dir'] is None:
        raise ValueError('Ground truth data not provided')

    # train / test
    classifier = None
    simsiam = None
    if mode == 'train':
        # datasets: {'train':(tr_data,tr_gt), 'val':(val_data,val_gt), 'test':(te_data, te_gt)}
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    augmentation_noisiness=args['augmentation_noisiness'],
                    data_split='tr-val',
                    tr_frac=args['tr_frac']
                )
        if args['trained_model_dir'] is None:
            trainer = train_simsiam(datasets, args)
            simsiam = trainer.model
        classifier = train_classifier(datasets['train'],datasets['val'],args,simsiam=simsiam)

    elif mode == 'test':
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    augmentation_noisiness=args['augmentation_noisiness'],
                    data_split='no',
                    tr_frac=args['tr_frac']
                )
        test_classifier(datasets['test'],args,classifier)

    elif mode == 'train_test':
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    augmentation_noisiness=args['augmentation_noisiness'],
                    data_split='tr-val-te',
                    tr_frac=args['tr_frac']
                )

        if args['trained_model_dir'] is None:
            trainer = train_simsiam(datasets, args)
            simsiam = trainer.model
        classifier = train_classifier(datasets['train'],datasets['val'],args,simsiam=simsiam)

        test_classifier(datasets['test'],args,classifier)
    else:
        ValueError('Please specify command arg -m or --mode to be train,test,or train_test')
