import sys
import pandas as pd
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

## aggregate to hourly at user level
# data_dir = '/nas/home/siyiguo/LA_tweets_emot_mf/2020'
# # month = sys.argv[1]
# files = glob(data_dir+'*.csv')

# """
# columns:
# created_at,id,text,user_id,user_screen_name,user_verified,user_followers_count,user_friends_count,user_favourites_count,
# anger,anticipation,disgust,fear,joy,love,optimism,pessimism,sadness,surprise,trust,
# care,harm,fairness,cheating,loyalty,betrayal,authority,subversion,purity,degradation
# """

# for file in files:
#     print(file)

#     df = pd.read_csv(file,lineterminator='\n')
#     df ['created_at'] = pd.to_datetime(df['created_at'])
#     df ['created_at'] = df ['created_at'].dt.tz_convert('US/Pacific')
#     user_attr = df.groupby([pd.Grouper(freq='H', key='created_at'),'user_screen_name'])['user_id','user_verified','user_followers_count','user_friends_count','user_favourites_count'].first()
#     agg_ts = df.groupby([pd.Grouper(freq='H', key='created_at'),'user_screen_name'])['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity', 'degradation'].sum()
#     agg_ts['count'] = df.groupby([pd.Grouper(freq='H', key='created_at'),'user_screen_name'])['id'].count()
#     pd.concat([user_attr,agg_ts],axis=1).reset_index().to_csv('/nas/home/siyiguo/user_similarity/data/LA_tweets_emot_mf_user_hourly_ts.csv',mode='a',index=False,header=False)


"""
final df index:
Index(['created_at', 'user_screen_name', 'user_id', 'user_verified',
       'user_followers_count', 'user_friends_count', 'user_favourites_count',
       'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
       'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'count'],
      dtype='object')
"""

## add header, some hours are split into 2 files, add them up
all_cols = ['time', 'user_screen_name', 'user_id', 'user_verified',
       'user_followers_count', 'user_friends_count', 'user_favourites_count',
       'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
       'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'count']
df = pd.read_csv('/nas/home/siyiguo/user_similarity/data/LA_tweets_emot_mf_user_hourly_ts.csv',names=all_cols,index_col=None)
print(df.shape)
user_attr = df.groupby(['time','user_screen_name'])['user_id','user_verified','user_followers_count','user_friends_count','user_favourites_count'].first()
agg_ts = df.groupby(['time','user_screen_name'])['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness', 'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity', 'degradation', 'count'].sum()
df_new = pd.concat([user_attr,agg_ts],axis=1).reset_index()
print(df_new.shape)

df_new.to_csv('/nas/home/siyiguo/user_similarity/data/LA_tweets_emot_mf_user_hourly_ts.csv',index=False)
