import pandas as pd
import numpy as np
import math
from glob import glob
import pickle
import os
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


"""
egyptuae: '2016-01-01' to '2019-08-05', 3D
venezuela: '2017-01-01' to '2021-06-03' 3D
cnhu: '2019-04-21' to '2021-04-05' 3D
"""

start_date = '2017-01-01'
end_date = '2021-06-03'
agg_time_period='3D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period,tz='utc')

nation = 'venezuela'

df = pd.read_csv(nation+'_all_controls_drivers.csv',lineterminator='\n')

# df['tweet_time'] = pd.to_datetime(df['tweet_time'],utc=True,format='mixed')
# logging.info(f"min date:{df['tweet_time'].min()}, max date:{df['tweet_time'].max()}")
# logging.info(f"languages:\n{df['tweet_language'].value_counts()}")
# logging.info(f"time distribution:\n{df['tweet_time'].dt.year.value_counts()}")
# logging.info(f"time zone check: {df['tweet_time'].apply(lambda t: t.tzinfo is None).sum()}, {pd.unique(df['tweet_time'].apply(lambda t: t.tzinfo))}")

# df = df[(df['tweet_time']>=pd.Timestamp(start_date,tz='utc')) & (df['tweet_time']<=pd.Timestamp(end_date,tz='utc'))]
# logging.info(f"data {start_date} to {end_date} {agg_time_period}: shape={df.shape}")

user_dic = df[['userid', 'user_screen_name']].drop_duplicates(subset=['user_screen_name']).set_index('user_screen_name').squeeze()
user_dic = user_dic.to_dict()
df['retweet_userid'] = df['retweet_username'].apply(lambda x: user_dic.get(x))

def fn_a(lst):
    return [dic[x] for x in lst if dic.get(x) is not None]

user_ts_data = df.groupby(['userid',pd.Grouper(freq=agg_time_period,key='tweet_time')])[['following_count','follower_count']].sum()
# user_ts_data['tweet_count'] = df.groupby(['userid',pd.Grouper(freq=agg_time_period,key='tweet_time')])['tweetid'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['userid','tweet_time']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}')

# keep record to make sure users are indexed in the same order
ordered_user_index = user_ts_data.groupby(level=0)[0].first().index

user_links = df.groupby(['userid'])['retweet_userid'].apply(list)
user_links = user_links.loc[ordered_user_index,]
dic = {u:i for i,u in enumerate(list(user_links.index))}
user_links = user_links.apply(fn_a)
max_links_len = user_links.apply(len).max()
logging.info(f"user links: max_links_len={max_links_len}, avg links len = {user_links.apply(len).mean()}, median links len={user_links.apply(len).median()}")
user_links_array = user_links.tolist()
user_links_array = np.array([(n+[-1]*max_links_len)[:max_links_len] for n in user_links_array])
logging.info(f'shape of np array for the user links data: {user_links_array.shape}')
pickle.dump(user_links_array, open(nation+'_links_data.pkl','wb'))