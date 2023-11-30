"""
Process LA data 
    ts_data: N*T*D, time series, N is number of users, T is len of time series (agg into daily), D is dimension of features
    demo_data: N*D_demo, demographic data, N is num users, D_demo is dimension of demographic features
    gt_data: N*1, the ground truth label of the clusters

"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import dateutil.tz
import logging
from tqdm import tqdm
import pickle

def create_logger():
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

create_logger()

feature_colnames = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
    'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
    'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
    'degradation', 'count'
]
demo_colnames = [
    'user_verified', 'user_followers_count', 'user_friends_count', 'user_favourites_count'
]

# load aggregated data (emotion/mfs per hour per user) for active users (20 or more tweets in 2020)
df = pd.read_csv('/nas/eclairnas01/users/siyiguo/LA_data/LA_tweets_emot_mf_active_user_hourly_ts.csv',lineterminator='\n')
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].apply(lambda x: x.astimezone(dateutil.tz.tzoffset(None, -28800))) # set all data to the same time zone
df = df[df['time']>=pd.Timestamp('2020-01-01',tz='US/Pacific')]
print(pd.Grouper(key='time',freq='D'))
print(f'len of data with active user more than 2 tweets in year of 2020: {len(df)}')

# aggregate to daily for now
user_ts_data = df.groupby(['user_screen_name',pd.Grouper(freq='D',key='time')])[feature_colnames].sum()
logging.info(f'raw user ts data - shape: {user_ts_data.shape}')

# transform into 3-d np array
ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
logging.info(f'shape of np array for the ts data: {ts_array.shape}')
pickle.dump(ts_array, open('/nas/eclairnas01/users/siyiguo/LA_data/LA_tweets_ts_data.pkl','wb'))
logging.info('finished saving ts data')

# build demographic data
demo_data = df.groupby(['user_screen_name'])[demo_colnames].first()
# make sure users are indexed in the same order
ordered_user_index = user_ts_data.groupby(level=0)['count'].first().index
demo_data = demo_data.loc[ordered_user_index,].values
logging.info(f'demographic data - shape: {demo_data.shape}')
pickle.dump(demo_data,open('/nas/eclairnas01/users/siyiguo/LA_data/LA_tweets_demo_data.pkl','wb'))
logging.info('finished saving demo data')

"""
df.columns
Index(['time', 'user_screen_name', 'user_id', 'user_verified',
       'user_followers_count', 'user_friends_count', 'user_favourites_count',
       'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
       'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'count'],
      dtype='object')
"""