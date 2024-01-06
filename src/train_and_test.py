import os
import time
import logging
import csv

from TSCluster import set_seed, get_training_args, create_logger
from TSCluster import read_data
from TSCluster import Trainer
from TSCluster import evaluate


def train(data, args):
    start_time = time.time()
    logging.info('Start training...')

    # set up trainer
    trainer = Trainer(data, args)

    trainer.train()

    logging.info(
        f"Finished training data. Time: {time.time()-start_time}"
    )

    return trainer


def test(data,args,trainer=None):
    start_time = time.time()

    logging.info('============ Evaluation on Test Data ============= \n')
    # evaluate with the best model just got from training or with the model given from config
    if trainer is not None:
        eval_model_path = args['output_dir'] + '/model_weights.h5'
    elif args.get('trained_model_dir') is not None:
        eval_model_path = args['trained_model_dir']
    else:
        raise ValueError('Please provide a model for evaluation.')

    evaluate(data, eval_model_path, args)

    # # output predictions
    # with open(args['output_dir'] + '/mf_preds.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(mf_preds)

    logging.info(
        f"Finished evaluating test data. Time: {time.time()-start_time}"
    )


if __name__ == '__main__':
    # logger
    create_logger()

    # args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    mode, args = get_training_args(root_dir)

    # set seed
    set_seed(args['seed'])

    # train / test
    trainer = None
    if mode == 'train':
        datasets = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    augmentation_noisiness=args['augmentation_noisiness'],
                    data_split='tr-val'
                )
        trainer = train(datasets, args)

    elif mode == 'test':
        datasets = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    augmentation_noisiness=args['augmentation_noisiness'],
                    data_split='no'
                )
        test(datasets,args,trainer)

    elif mode == 'train_test':
        datasets = read_data(
                    ts_data_dir=args['ts_data_dir'],
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    augmentation_noisiness=args['augmentation_noisiness'],
                    data_split='no'#'tr-val-te'
                )

        trainer = train(datasets, args)
        test(datasets,args,trainer)
    else:
        ValueError('Please specify command arg -m or --mode to be train,test,or train_test')
