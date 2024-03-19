"""Utility functions"""

import os
import random
from datetime import datetime
import logging
import argparse
import numpy as np
import tensorflow as tf


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def create_logger(args=None):
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    if args is not None:
        logfilename = os.path.join(args['output_dir'],logfilename)
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True


def read_command_args(args,root_dir):
    """Read arguments from command line."""

    parser = argparse.ArgumentParser(description='Unsupervised Time Series Clustering.')
    parser.add_argument('-m', '--mode', type=str, required=False, default='train_test', help='train,test,or train_test')
    parser.add_argument('-c', '--config_dir', type=str, required=False, default=None, help='configuration file dir that specifies hyperparameters etc')
    parser.add_argument('-i', '--data_dir', type=str, required=True, help='input data directory')
    parser.add_argument('-d', '--demo_data_dir', type=str, required=False, default=None, help='demographic data directory')
    parser.add_argument('-g', '--gt_dir', type=str, required=False, default=None, help='ground truth label directory')
    parser.add_argument('-o', '--output_dir', type=str, required=False, default='./output', help='output directory')
    parser.add_argument('-t', '--trained_model', type=str, required=False, default=None, help='if testing, it is optional to provide a trained model weight dir')
    parser.add_argument('-n', '--augmentation_noisiness', type=float, required=False, default=0.3, help='how much noise to inject when performing positive example augmentation for contrastive learning')
    parser.add_argument('-l', '--max_triplet_len', type=int, required=False, default=1000, help='how much noise to inject when performing positive example augmentation for contrastive learning')
    parser.add_argument('-s', '--seed', type=int, required=False, default=3, help='random seed')
    parser.add_argument('--temperature', type=float, required=False, default=0.2, help='temperature scaling')
    parser.add_argument('--trained_classifier_model', type=str, required=False, default=None, help='if testing, it is optional to provide a trained classifier model weight dir')
    parser.add_argument('--tau', type=float, required=False, default=1.0, help='controls how much timestamp contributes to the final representation')
    parser.add_argument('--lam', type=float, required=False, default=1.0, help='controls how much timestamp textual features to the final representation')

    command_args = parser.parse_args()

    # mode
    mode = command_args.mode

    # data dir
    # root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    args['ts_data_dir'] = os.path.join(root_dir, command_args.data_dir)
    args['demo_data_dir'] = os.path.join(root_dir, command_args.demo_data_dir) if command_args.demo_data_dir else None
    args['gt_dir'] = os.path.join(root_dir, command_args.gt_dir) if command_args.gt_dir else None
    args['config_dir'] = os.path.join(root_dir, command_args.config_dir) if command_args.config_dir else None
    args['trained_model_dir'] = os.path.join(root_dir, command_args.trained_model) if command_args.trained_model else None
    args['trained_classifier_model_dir'] = os.path.join(root_dir, command_args.trained_classifier_model) if command_args.trained_classifier_model else None
    args['output_dir'] = os.path.join(root_dir, command_args.output_dir)
    if not os.path.exists(os.path.join(root_dir, args['output_dir'])):
        os.makedirs(os.path.join(root_dir, args['output_dir']))
    args['augmentation_noisiness'] = float(command_args.augmentation_noisiness)
    args['max_triplet_len'] = int(command_args.max_triplet_len)
    args['temperature'] = float(command_args.temperature)
    args['seed'] = int(command_args.seed)
        
    return mode, args


def read_config(args):
    """
    Read arguments from the config file.

    Args:
        args: a dict to store arguments, should at least include 'config_dir'
    """
    # default values
    args['lr'] = 0.0005
    args['batch_size'] = 64
    args['epoch'] = 40
    args['loss_fn'] = 'InfoNCE' #'SimSiam' #
    args['patience'] = 10
    args['weight_decay'] = 0.0005
    args['embed_dim'] = 50
    args['n_transformer_layer'] = 2
    args['n_attn_head'] = 4
    args['binarize_gt'] = True
    args['dropout'] = 0.3
    args['n_cluster'] = 10
    args['tr_frac'] = 0.8
    
    # read args in config file
    if args['config_dir'] is not None:
        with open(args['config_dir'], 'r') as f:
            for l in f.readlines():
                # skip comments
                if l.strip() == '' or l.strip().startswith('#'):
                    continue
                # get value
                arg = l.strip().split(" = ")
                arg_name, arg_val = arg[0], arg[1]

                args[arg_name] = arg_val

        # hyperparameters
        args['lr'] = float(args['lr'])
        args['batch_size'] = int(args['batch_size'])
        args['epoch'] = int(args['epoch'])
        args['loss_fn'] = str(args['loss_fn'])
        args['patience'] = int(args['patience'])
        args['weight_decay'] = float(args['weight_decay'])
        args['embed_dim'] = int(args['embed_dim'])
        args['n_transformer_layer'] = int(args['n_transformer_layer'])
        args['n_attn_head'] = int(args['n_attn_head'])
        args['max_triplet_len'] = int(args['max_triplet_len'])
        args['augmentation_noisiness'] = float(args['augmentation_noisiness'])
        args['binarize_gt'] = bool(args['binarize_gt'])
        args['dropout'] = float(args['dropout'])
        args['n_cluster'] = int(args['n_cluster'])
        args['tr_frac'] = float(args['tr_frac'])
        args['seed'] = int(args['seed'])

    return args

def get_training_args(root_dir):
    args = {}
    mode, args = read_command_args(args,root_dir)
    args = read_config(args)

    return mode, args