import os
import logging
import numpy as np
import tensorflow as tf

from .model import SimSiam
from TSCluster import create_dataset

# TODO: implement extra callback to check labeled data
# class CustomCallback(Callback):
#     def __init__(self, validation_data, batch_size):
#         self.val_x, self.val_y = validation_data
#         self.batch_size = batch_size
#         super(Callback, self).__init__()

#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = self.model.predict(self.val_x, verbose=0, batch_size=self.batch_size)
#         if type(y_pred)==type([]):
#             y_pred = y_pred[0]
#         precision, recall, thresholds = precision_recall_curve(self.val_y, y_pred)
#         pr_auc = auc(recall, precision)
#         roc_auc = roc_auc_score(self.val_y, y_pred)
#         logs['custom_metric'] = pr_auc + roc_auc
#         print ('val_aucs:', pr_auc, roc_auc)


class Trainer:
    """
    Args:
        datasets: {'train':((tr_data,tr_aug_data),tr_gt), 'val':((val_data,val_aug_data),val_gt), 'test':((te_data,te_aug_data),te_gt)} 
                  tr_data is [demo,timestamp_array,values_array,feat_dummy_array], 
                  each of these array shape (N * max_triplet_len)
        args: dict of parameters
    """
    def __init__(self,datasets,args):
        self.tr_X = datasets['train'][0] # pair (tr_data, aug_data)
        self.tr_y = datasets['train'][1] # could be None
        self.val_X = datasets['val'][0]
        self.val_y = datasets['val'][1]
        # number of data points
        self.N = len(self.tr_X[0][1])

        self.args = args

        self.model = SimSiam(self.args)
    
    def create_scheduler(self):
       n_steps = self.args['epoch'] * (self.N // self.args['batch_size'])
       self.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.args['lr'], decay_steps=n_steps
            )
       
    def create_callbacks(self):
        self.es = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=self.args['patience'], restore_best_weights=True
            )
        
        self.callbacks = [self.es]

    def train(self,savepath=None):
        # self.create_scheduler()
        self.create_callbacks()

        self.tr_X = create_dataset(self.tr_X, self.args['batch_size'])

        # Compile model and start training.
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.args['lr'])) #(self.lr_decayed_fn))
        
        history = self.model.fit(
            self.tr_X, 
            # batch_size=self.args['batch_size'], 
            epochs=self.args['epoch'], 
            # validation_data=(self.val_X),
            callbacks=self.callbacks
        )
        if savepath is None: savepath = self.args['output_dir']+'/model_weights.h5'
        self.model.save_weights(savepath)

        logging.debug('Finished training SimSiam. Log:')
        logging.debug(history.history)