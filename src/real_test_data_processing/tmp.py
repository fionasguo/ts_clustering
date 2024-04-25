import pandas as pd
import numpy as np
import math
import re
from glob import glob
import pickle
import os, sys
from datetime import datetime
from ast import literal_eval
import logging
from tqdm import tqdm


def create_logger():
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True

create_logger()



data_dir = '/nas/eclairnas01/users/siyiguo/hashed_infoOps/'
nation = str(sys.argv[1])
i = str(sys.argv[2])
path = data_dir+nation+'/all_controls_'+i+'.csv'
logging.info(path)

control = pd.read_csv(path, lineterminator='\n')

control = control.rename(columns={'tweet_hashtags':'hashtags'})
control = control.reset_index(drop=True)
logging.info(f"control shape={control.shape}\n{control.columns}")

def fn_a(row):
    try:
        tmp = literal_eval(row['data_referenced_tweets'])
        for i in tmp:
            if i['type'] == 'retweeted':
                row['retweet_tweetid'] = i['id']
                row['is_retweet'] = True
                
                match = re.search(r'RT @([A-Za-z0-9_]+)',row['tweet_text'])
                if match:
                    row['retweet_username'] = match.group(1)
                else:
                    logging.info(f"{row.name} {row['data_referenced_tweets']}  {row['tweet_text']}")
                    row['retweet_username'] = None
    except:
        pass
    return row

tqdm.pandas()

control[['is_retweet','retweet_userid', 'retweet_tweetid', 'retweet_username']] = None # wrong from keith's data
control = control.progress_apply(fn_a,axis=1)

logging.info(f"agg control file: shape={control.shape}, min date:{control['tweet_time'].min()}, max date:{control['tweet_time'].max()}, num users: {len(pd.unique(control['userid']))}\n{control.columns}")
logging.info(f"is_retweet={control['is_retweet'].sum()}, retweet_userid={len(control) - control['retweet_userid'].isnull().sum()}, retweet_username={len(control) - control['retweet_username'].isnull().sum()}")

control.to_csv(path+'.prc',index=False)