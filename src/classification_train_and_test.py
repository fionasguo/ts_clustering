import os
import time
import logging
import pickle
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import losses, Model
from tensorflow.keras.layers import Dense, ReLU, Dropout
from tensorflow.keras.utils import to_categorical

from TSCluster import set_seed, get_training_args, create_logger
from TSCluster import read_data
from TSCluster import SimSiam, Trainer
from TSCluster import plot_tsne


def create_classification_dataset(data_tuple,batch_size):
    """
    create tf.data.Dataset from np arrays
    data_tuple: ((ts_data,aug_data), (indices,links), y)
        ts_data: list of lists [demo,timestamp_array,values_array,feat_dummy_array], each of these array shape (N * max_triplet_len)
    """
    X,_, y = data_tuple
    ts_data = X[0]
    ts_data = {
        'demo':ts_data[0],
        'timestamps':ts_data[1],
        'values':ts_data[2],
        'feat':ts_data[3]
    }

    if len(np.unique(y)) > 2:
        # one-hot
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        # convert integers to dummy variables (i.e. one hot encoded)
        y = to_categorical(y)

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
    activation = 'sigmoid' if n_classes==1 else "softmax"

    # x = simsiam.predictor(simsiam.encoder.output)
    x = Dense(args['embed_dim'],activation='relu')(simsiam.encoder.output)
    x = Dropout(args['dropout'])(x)
    x = Dense(n_classes,activation=activation)(x)

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

    classifier.compile(
        loss=loss_fn,
        optimizer='adam',
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.F1Score(average=None,threshold=0.2,name="F1-02"),
            tf.keras.metrics.F1Score(average=None,threshold=0.3,name="F1-03"),
            tf.keras.metrics.F1Score(average=None,threshold=0.35,name="F1-035"),
            tf.keras.metrics.F1Score(average=None,threshold=0.4,name="F1-04"),
            tf.keras.metrics.F1Score(average=None,threshold=0.45,name="F1-045"),
            tf.keras.metrics.F1Score(average=None,threshold=0.5,name="F1-05"),
            tf.keras.metrics.F1Score(average=None,threshold=0.6,name="F1-06"),
            tf.keras.metrics.F1Score(average=None,threshold=0.7,name="F1-07"),
            tf.keras.metrics.F1Score(average=None,threshold=0.8,name="F1-08"),
            ]
    )
    
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
    pickle.dump(tr_emb,open(args['output_dir']+'/classifier_train_embeddings.pkl','wb'))
    tr_labels = np.asarray(list(train_dataset.map(lambda X,y:y)))
    if tr_labels.shape[-1] == 1:
        tr_labels = tr_labels.flatten()
    else:
        tr_labels = tr_labels.reshape((-1,tr_labels.shape[-1]))
        tr_labels = np.argmax(tr_labels,axis=1).flatten()
    val_emb = embedding_model.predict(val_dataset.map(lambda X, y: X))
    val_labels = np.asarray(list(val_dataset.map(lambda X,y:y)))
    if val_labels.shape[-1] == 1:
        val_labels = val_labels.flatten()
    else:
        val_labels = val_labels.reshape((-1,val_labels.shape[-1]))
        val_labels = np.argmax(val_labels,axis=1).flatten()

    plot_tsne(tr_emb, tr_labels, fig_save_path=args['output_dir']+'/classifier_train_data_')
    plot_tsne(val_emb, val_labels, fig_save_path=args['output_dir']+'/classifier_val_data_')

    return classifier

def test_classifier(test_data, args, classifier):
    logging.info("Testing Classifier")
    if classifier is None: 
        if args.get('trained_classifier_model_dir') is not None:
            eval_model_path = args['trained_classifier_model_dir']
            classifier = build_classifier(args,simsiam=None)
            classifier.built = True
            classifier.load_weights(eval_model_path)

            if args['n_classes'] == 2:
                loss_fn = losses.BinaryCrossentropy(from_logits=False)
            else:
                loss_fn = losses.CategoricalCrossentropy(from_logits=False)

            classifier.compile(
                loss=loss_fn,
                optimizer='adam',
                metrics=[
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.F1Score(average=None,threshold=0.2,name="F1-02"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.3,name="F1-03"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.35,name="F1-035"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.4,name="F1-04"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.45,name="F1-045"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.5,name="F1-05"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.6,name="F1-06"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.7,name="F1-07"),
                    tf.keras.metrics.F1Score(average=None,threshold=0.8,name="F1-08"),
                    ]
            )
            logging.info('Classifier loaded')
            classifier.summary(print_fn=logging.info)
        else:
            raise ValueError('Please provide a model for classification evaluation.')

    test_dataset = create_classification_dataset(test_data, args['batch_size'])

    eval_results = classifier.evaluate(test_dataset,batch_size=args['batch_size'])
    test_preds_conf = classifier.predict(test_dataset,batch_size=args['batch_size'])
    pickle.dump(test_preds_conf,open(args['output_dir']+'/classifier_test_preds_conf.pkl','wb'))

    # plot tsne
    embedding_model = Model(inputs=classifier.input, outputs=classifier.get_layer('embed_output').output)
    test_emb = embedding_model.predict(test_dataset.map(lambda X, y: X))
    pickle.dump(test_emb,open(args['output_dir']+'/classifier_test_embeddings.pkl','wb'))
    test_labels = np.asarray(list(test_dataset.map(lambda X,y:y)))
    if test_labels.shape[-1] == 1:
        test_labels = test_labels.flatten()
    else:
        test_labels = test_labels.reshape((-1,test_labels.shape[-1]))
        test_labels = np.argmax(test_labels,axis=1).flatten()
    pickle.dump(test_labels,open(args['output_dir']+'/classifier_test_gt_labels.pkl','wb'))
    plot_tsne(test_emb, test_labels, fig_save_path=args['output_dir']+'/classifier_test_data_')

    logging.info(f"Finished evaluating test data.\nLoss: {eval_results[0]}, Precision:{eval_results[1]}, Recall:{eval_results[2]}, AUC-ROC:{eval_results[3]}, F1 score:{eval_results[4]} {eval_results[5]} {eval_results[6]} {eval_results[7]} {eval_results[8]} {eval_results[9]} {eval_results[10]} {eval_results[11]} {eval_results[12]}")
    
    if args['n_classes'] == 2:
        test_preds_conf = test_preds_conf.flatten()
        test_preds = np.zeros(shape=test_preds_conf.shape)
        test_preds[test_preds_conf>=0.5] = 1
        logging.info(f"manual - AUC={roc_auc_score(test_labels,test_preds)}")
    else:
        test_preds = np.argmax(test_preds_conf,axis=1)
    logging.info(classification_report(test_labels,test_preds))

    return eval_results


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
    simsiam = None
    if mode == 'train':
        # datasets: {'train':(tr_data,tr_gt), 'val':(val_data,val_gt), 'test':(te_data, te_gt)}
        datasets, args = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    args=args,
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    links_data_dir=args['links_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    data_aug=False,
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
                    links_data_dir=args['links_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    data_aug=False,
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
                    links_data_dir=args['links_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    data_aug=False,
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
