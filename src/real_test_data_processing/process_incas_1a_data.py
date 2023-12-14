"""
Process INCAS phase1a data 
    ts_data: N*T*D, time series, N is number of users, T is len of time series (agg into 12H), D is dimension of features
    demo_data: N*D_demo, demographic data, N is num users, D_demo is dimension of demographic features
    gt_data: N*1, the ground truth label of the clusters

"""

import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx
import umap
import re
import pickle
import os
from datetime import datetime
import logging
import itertools
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_logger():
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True


def find_top_hashtags(df_series,k,pattern = r'#\S+'):
    """find top k hashtags from df_series wihich is a column of text"""
    hashtags = [x for text in df_series.values for x in re.findall(pattern, text)]
    hashtags = [(i, count, round(count/len(hashtags)*100.0, 2)) for i, count in Counter(hashtags).most_common(k)]
    return hashtags


create_logger()

user_tot_tweet_threshold = 10 # only take users who has more than 10 tweets in about 90 days
start_date = '2017-03-01'
end_date = '2017-06-01'
agg_time_period='12H'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2048
MODEL = 'Yanzhu/bertweetfr-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 5

demo_colnames = ['followingCount', 'followerCount']
feature_colnames = [
    'agenda-1.4_Believe that ENTITY or GROUP is moral/ethical/honest/beneficial',
    'agenda-1.3_Believe that ENTITY or GROUP is immoral/unethical/dishonest/harmful',
    'agenda-2.2.1_vote for',  'agenda-2.2.2_vote against', 
    'Democracy', 'Economy', 'Environment and Climate Change', 'Fake News/Misinformation', 
    'Immigration and Refugees', 'International Alliance Organizations',
    'National Identity and National Pride', 'Relationship with Russia',
    'Religion', 'Terrorism and Counterterrorism',
    'emotion-4.1_Anger, Hate, Contempt, Disgust',
    'emotion-4.2_Embarrassment, Guilt, Shame, Sadness',
    'emotion-4.3_Admiration, Love', 'emotion-4.4_Optimism, Hope',
    'emotion-4.5_Joy, Happiness', 'emotion-4.6_Pride, including National Pride',
    'emotion-4.7_Fear, Pessimism', 'emotion-4.8_Amusement',
    'emotion-4.9_Positive-other', 'emotion-4.10_Negative-other'
]

logging.info('start reading data')
df = pd.read_csv('/nas/eclairnas01/users/siyiguo/incas_data/phase1a.csv',lineterminator='\n') # data is 8.25 GB
logging.info('formating timestamps in the data')
# format timestamps
df['timePublished'] = pd.to_datetime(pd.to_datetime(df['timePublished']).astype(int),unit='ms') # pd.to_datetime(df['timePublished'],unit='ms')
logging.info(f'raw data shape: {df.shape}')
# get all data between 2017/03/01 to 05/31
df = df[(df['timePublished']>=pd.Timestamp(start_date)) & (df['timePublished']<=pd.Timestamp(end_date))]
logging.info(f'raw data between 03/01 to 05/31: {df.shape}')
# only take users with more than some tweets
active_users = df.groupby(['twitterAuthorScreenname'])['id'].count()
tot_user_num = len(active_users)
active_users = active_users[active_users>user_tot_tweet_threshold]
active_user_set = set(active_users.index)
df = df[df['twitterAuthorScreenname'].isin(active_user_set)]
df = df.reset_index(drop=True)
logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/tot_user_num}')
logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')

######################## affect features ########################
# the sum of prob of each feature during 12H period for each user
user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[feature_colnames].sum()
user_ts_data['tweet_count'] = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
logging.info(f'raw user affect feature ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(active_user_set)}, len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}')
# pickle.dump(ts_array, open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_affect_ts_data.pkl','wb'))
# logging.info('finished saving affect ts data')

######################## BERT embedding features ########################
# ts data with another set of features - bert embedding - umap reduced
# logging.info('start computing BERT embeddings')
# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     encoded_input = tokenizer(tmp['contentText'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
# logging.info(f'BERT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings.pkl','wb'))
# logging.info('BERT embeddings saved.')
# all_embeddings = pickle.load(open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings.pkl','rb'))
# logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')
# # # dim reduction - UMAP - OOM
# # reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',verbose=True)
# # all_embeddings = reducer.fit_transform(all_embeddings)
# # logging.info(f'UMAP finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# # dim reduction - pca
# logging.info('start PCA')
# all_embeddings = StandardScaler().fit_transform(all_embeddings)
# reducer = PCA(n_components=n_comp)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# df[list(range(n_comp))] = all_embeddings
# user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[list(range(n_comp))].sum()
# user_ts_data['tweet_count'] = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
# logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# # fill the time series with the entire time range
# user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
# logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(active_user_set)}, len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
# pickle.dump(ts_array, open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings_ts_data.pkl','wb'))
# logging.info('finished saving BERT embeddings ts data')

######################## hashtag features ########################
# logging.info('start building hashtag features')
# # find hashtags for each user
# tqdm.pandas()
# df['hashtags'] = df['contentText'].progress_apply(lambda text: re.findall(r'#\S+', text)) # a list
# logging.info(f"done detecting hashtags for individual tweets, avg # hashtags for individual user: {df['hashtags'].apply(len).mean()}")
# logging.info("getting top 50 hashtags in all text'")
# # find top 50 hashtags in all text
# all_hashtags = [hashtag for user_hashtags in df['hashtags'].tolist() for hashtag in user_hashtags]
# logging.info(f'number of all hashtags found in the data: {len(all_hashtags)}')
# all_hashtags = [h[0] for h in Counter(all_hashtags).most_common(50)]
# all_hashtags_set = set(all_hashtags)
# logging.info(f"top 50 hashtags extracted")
# # all_hashtags = [(k, sum(amt for _,amt in v)) for k,v in itertools.groupby(sorted(all_hashtags), key=lambda tup: tup[0])]
# # all_hashtags = sorted(all_hashtags,key=lambda tup: tup[1],reverse=True)[:50]
# # all_hashtags = [h[0] for h in all_hashtags]

# onehot_hashtags = [[1 if h in user_hashtags else 0 for h in all_hashtags] for user_hashtags in df['hashtags'].apply(set).tolist()]
# df[all_hashtags] = onehot_hashtags
# logging.info(f'one hot encoding done, top 50 hashtags and their freq:\n{df[all_hashtags].mean(axis=0)}')
# # aggregate to user ts data
# user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[all_hashtags].sum()
# user_ts_data['twitter_count'] = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
# logging.info(f'raw user hashtag ts data - shape: {user_ts_data.shape}')
# # fill the time series with the entire time range
# user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
# logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(active_user_set)}, len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}')
# pickle.dump(ts_array, open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_hashtag_ts_data.pkl','wb'))
# logging.info('finished saving hashtag ts data')

######################## demographic data ########################
# # build demographic data
# demo_data = df.groupby(['twitterAuthorScreenname'])[demo_colnames].first()
# # make sure users are indexed in the same order
ordered_user_index = user_ts_data.groupby(level=0)['tweet_count'].first().index
# demo_data = demo_data.loc[ordered_user_index,].values
# logging.info(f'demographic data - shape: {demo_data.shape}')
# pickle.dump(demo_data,open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_demo_data.pkl','wb'))
# logging.info('finished saving demo data')

######################## ground truth data - hashtag_coord ########################
# # get ground truth data
# gt_data_file = '/nas/eclairnas01/users/siyiguo/incas_data/hashtag_coord_phase1a.edgelist'
# G = nx.read_edgelist(gt_data_file)

# df['contentText'] = df['contentText'].str.replace(',','')
# df['contentText'] = df['contentText'].str.replace('.','')
# df['contentText'] = df['contentText'].str.replace(';','')
# df_user = df.groupby('twitterAuthorScreenname')

# gt_data = pd.DataFrame(index=ordered_user_index,columns=['label','hashtags'])
# label = 0
# for c in nx.connected_components(G): # c is a set of user names
#     df_cluster = []
#     for u in c:
#         if u in active_user_set:
#             df_cluster.append(df_user.get_group(u))
#     if len(df_cluster) == 0: continue
#     df_cluster = pd.concat(df_cluster)
#     # df_cluster =pd.concat([df_user.get_group(u) for u in c])
#     hashtags = find_top_hashtags(df_cluster['contentText'],10)
#     gt_data.loc[gt_data.index.isin(c),'label'] = label
#     gt_data.loc[gt_data.index.isin(c),'hashtags'] = gt_data.loc[gt_data.index.isin(c),'hashtags'].apply(lambda x: hashtags)
#     label += 1
# logging.info(f"gt data df shape: {gt_data.shape}, number of clusters: {label+1} (last cluster is unknown), number of users with identified clusters: {gt_data['label'].count()}")
# gt_data['label'] = gt_data['label'].fillna(label)

# # find the top hashtag in tweet
# gt_data['ind_hashtag'] = df.groupby('twitterAuthorScreenname')['contentText'].apply(lambda x: find_top_hashtags(x,5))

# pickle.dump(gt_data['label'].values,open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_gt_data.pkl','wb'))
# gt_data.to_csv('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_gt_data_df.csv')
# logging.info('finished saving ground truth data')

######################## ground truth data - time_coord ########################
# gt_data_file = '/nas/eclairnas01/users/siyiguo/incas_data/time_coord_phase1a.edgelist'
# G = nx.read_edgelist(gt_data_file)

# df['contentText'] = df['contentText'].str.replace(',','')
# df['contentText'] = df['contentText'].str.replace('.','')
# df['contentText'] = df['contentText'].str.replace(';','')
# df_user = df.groupby('twitterAuthorScreenname')

# gt_data = pd.DataFrame(index=ordered_user_index,columns=['label','hashtags'])
# label = 0
# for c in nx.connected_components(G): # c is a set of user names
#     df_cluster = []
#     for u in c:
#         if u in active_user_set:
#             df_cluster.append(df_user.get_group(u))
#     if len(df_cluster) == 0: continue
#     df_cluster = pd.concat(df_cluster)
#     # df_cluster =pd.concat([df_user.get_group(u) for u in c])
#     hashtags = find_top_hashtags(df_cluster['contentText'],10)
#     gt_data.loc[gt_data.index.isin(c),'label'] = label
#     gt_data.loc[gt_data.index.isin(c),'hashtags'] = gt_data.loc[gt_data.index.isin(c),'hashtags'].apply(lambda x: hashtags)
#     label += 1
# logging.info(f"gt data df shape: {gt_data.shape}, number of clusters: {label+1} (last cluster is unknown), number of users with identified clusters: {gt_data['label'].count()}")
# gt_data['label'] = gt_data['label'].fillna(label)

# # find the top hashtag in tweet
# gt_data['ind_hashtag'] = df.groupby('twitterAuthorScreenname')['contentText'].apply(lambda x: find_top_hashtags(x,5))

# pickle.dump(gt_data['label'].values,open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_time_coord_gt_data.pkl','wb'))
# gt_data.to_csv('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_time_coord_gt_data_df.csv')
# logging.info('finished saving ground truth data')

######################## ground truth data - retweet_coord ########################
gt_data_file = '/nas/eclairnas01/users/siyiguo/incas_data/retweet_coord_phase1a.edgelist'
G = nx.read_edgelist(gt_data_file)

df['contentText'] = df['contentText'].str.replace(',','')
df['contentText'] = df['contentText'].str.replace('.','')
df['contentText'] = df['contentText'].str.replace(';','')
df_user = df.groupby('twitterAuthorScreenname')

gt_data = pd.DataFrame(index=ordered_user_index,columns=['label','hashtags'])
label = 0
for c in nx.connected_components(G): # c is a set of user names
    df_cluster = []
    for u in c:
        if u in active_user_set:
            df_cluster.append(df_user.get_group(u))
    if len(df_cluster) == 0: continue
    df_cluster = pd.concat(df_cluster)
    # df_cluster =pd.concat([df_user.get_group(u) for u in c])
    hashtags = find_top_hashtags(df_cluster['contentText'],10)
    gt_data.loc[gt_data.index.isin(c),'label'] = label
    gt_data.loc[gt_data.index.isin(c),'hashtags'] = gt_data.loc[gt_data.index.isin(c),'hashtags'].apply(lambda x: hashtags)
    label += 1
logging.info(f"gt data df shape: {gt_data.shape}, number of clusters: {label+1} (last cluster is unknown), number of users with identified clusters: {gt_data['label'].count()}")
gt_data['label'] = gt_data['label'].fillna(label)

# find the top hashtag in tweet
gt_data['ind_hashtag'] = df.groupby('twitterAuthorScreenname')['contentText'].apply(lambda x: find_top_hashtags(x,5))

pickle.dump(gt_data['label'].values,open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_retweet_coord_gt_data.pkl','wb'))
gt_data.to_csv('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_retweet_coord_gt_data_df.csv')
logging.info('finished saving ground truth data')









"""
df.columns
Index(['id', 'name', 'contentText', 'mediaType', 'language', 'imageUrls',
       'timePublished', 'retweetCount', 'twitterAuthorScreenname', 'tweetId',
       'twitterUserId', 'followingCount', 'followerCount', 'engagementType',
       'engagementParentId', 'mentionedUsers', 'providerName', 'tweetId.1',
       'tweetName', 'time_dt', 'date', 'tweet_duplicates',
       'retweet_duplicates', 'reply_duplicates', 'user', 'num_pro_macron',
       'num_pro_MLP', 'tweet_number_tweets', 'tweet_number_high_at_tweets',
       'tweet_number_unique_tweets', 'reply_number_tweets',
       'reply_number_high_at_tweets', 'reply_number_unique_tweets',
       'retweet_number_tweets', 'retweet_number_high_at_tweets',
       'retweet_number_unique_tweets', 'number_unique_tweets',
       'tweet_FractRepeated', 'reply_FractRepeated', 'retweet_FractRepeated',
       'Fract_MLP', 'MLPMacronOdds',
       'agenda-1.4_Believe that ENTITY or GROUP is moral/ethical/honest/beneficial',
       'agenda-1.3_Believe that ENTITY or GROUP is immoral/unethical/dishonest/harmful',
       'agenda-2.2.1_vote for', 'agenda-2.2.2_vote against', 'Democracy',
       'Economy', 'Environment and Climate Change', 'Fake News/Misinformation',
       'Immigration and Refugees', 'International Alliance Organizations',
       'National Identity and National Pride', 'Relationship with Russia',
       'Religion', 'Terrorism and Counterterrorism', 'Unnamed: 0',
       'emotion-4.1_Anger, Hate, Contempt, Disgust',
       'emotion-4.2_Embarrassment, Guilt, Shame, Sadness',
       'emotion-4.3_Admiration, Love', 'emotion-4.4_Optimism, Hope',
       'emotion-4.5_Joy, Happiness',
       'emotion-4.6_Pride, including National Pride',
       'emotion-4.7_Fear, Pessimism', 'emotion-4.8_Amusement',
       'emotion-4.9_Positive-other', 'emotion-4.10_Negative-other'],
      dtype='object')
"""