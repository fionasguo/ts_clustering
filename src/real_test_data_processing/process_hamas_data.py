import logging
import pandas as pd


import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2048
MODEL = 'sentence-transformers/stsb-xlm-r-multilingual' # 'cardiffnlp/twitter-xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 5

user_tot_tweet_threshold = 5

start_date = '2023-09-01' #'2009-07-06'
end_date = '2023-11-02'
agg_time_period='12H'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period,tz='utc')

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

data_dir = '/nas/eclairnas01/users/siyiguo/hamas_data/'

# df = pd.read_json(data_dir+'challenge_problem_two_21NOV.jsonl',lines=True)
# logging.info(f"raw jsonl file shape {df.shape}, columns:\n{df.columns}")
# df = df[['id','timePublished','contentText','language','mediaType','mediaTypeAttributes']]
# tmp = df['mediaTypeAttributes'].apply(pd.Series)
# tmp_twitter = tmp['twitterData'].apply(pd.Series)
# tmp_twitter = tmp_twitter[['engagementType','followerCount','followingCount','tweetId', 'twitterAuthorScreenname']]
# df = pd.concat([df.drop(columns=['mediaTypeAttributes']),tmp_twitter],axis=1)
# df = df[df.mediaType=='Twitter']
# df = df.drop(columns=['mediaType'])

# df.to_csv(data_dir+'all_hamas_twitter_data.csv',index=False)
# logging.info("all data csv saved")

df = pd.read_csv(data_dir+'all_hamas_twitter_data.csv',lineterminator='\n')
logging.info('csv file loaded')

df['timePublished'] = pd.to_datetime(df.timePublished, unit='ms', utc=True)
logging.info(f"data shape={df.shape}, columns:\n{df.columns}")
logging.info(f"min date: {df['timePublished'].min()}, max date: {df['timePublished'].max()}")
logging.info(f"language:\n{df['language'].value_counts()}")

user_tweet_count = df.groupby('twitterAuthorScreenname')['id'].count()
logging.info(f"total num of users: {len(user_tweet_count)}")
logging.info(f"num tweets per user: mean={user_tweet_count.mean()}, std={user_tweet_count.std()}")
logging.info(f"0.5 quantile={user_tweet_count.quantile(q=0.5)}, 0.75 quantile={user_tweet_count.quantile(q=0.75)}, 0.8 quantile={user_tweet_count.quantile(q=0.8)}, 0.9 quantile={user_tweet_count.quantile(q=0.9)}")

user_ts_count = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
logging.info('\n\n')

# check luca's gt
with open(data_dir+'gaza_coordinated_accounts.txt','r') as f:
    coord_users = f.read().splitlines()
coord_users = set(coord_users)
all_users = set(list(user_tweet_count.index))
logging.info(f"all users={len(all_users)}, luca's coord users={len(coord_users)}, intersection={len(all_users.intersection(coord_users))}")

active_users = user_tweet_count[user_tweet_count>=user_tot_tweet_threshold]
active_user_set = set(active_users.index)
df = df[df['twitterAuthorScreenname'].isin(active_user_set)]
df = df.reset_index(drop=True)
logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/len(user_tweet_count)}')
logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')
logging.info(f"active users={len(active_user_set)}, luca's coord users={len(coord_users)}, intersection={len(active_user_set.intersection(coord_users))}")


"""
df.columns
Index(['id', 'timePublished', 'contentText', 'language', 'engagementType',
       'followerCount', 'followingCount', 'tweetId',
       'twitterAuthorScreenname'],
      dtype='object')

"""

######################## BERT embedding features ########################
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
# pickle.dump(all_embeddings,open(data_dir+'bert_embeddings.pkl','wb'))
# logging.info('BERT embeddings saved.')
# # all_embeddings = pickle.load(open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings.pkl','rb'))
# # logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')

# # dim reduction - pca
# logging.info('start PCA')
# all_embeddings = StandardScaler().fit_transform(all_embeddings)
# reducer = PCA(n_components=n_comp)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'bert_embeddings_pca.pkl','wb'))
# logging.info('PCA saved.')

all_embeddings = pickle.load(open(data_dir+'bert_embeddings_pca.pkl','rb'))

df[list(range(n_comp))] = all_embeddings
user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[list(range(n_comp))].sum()
# user_ts_data['tweet_count'] = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(active_user_set)}, len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
# pickle.dump(ts_array, open(data_dir+'bert_embeddings_ts_data.pkl','wb'))
# logging.info('finished saving BERT embeddings ts data')

ordered_user_index = user_ts_data.groupby(level=0)[0].first().index

######################## demographic data ########################
# # build demographic data
# demo_colnames = ['followerCount', 'followingCount']
# demo_data = df.groupby(['twitterAuthorScreenname'])[demo_colnames].first()
# # make sure users are indexed in the same order

# demo_data = demo_data.loc[ordered_user_index,].values
# logging.info(f'demographic data - shape: {demo_data.shape}')
# pickle.dump(demo_data,open(data_dir+'demo_data.pkl','wb'))
# logging.info('finished saving demo data')

# ####################### ground truth data - hashtag_coord ########################
# get ground truth data
# coord_users = pickle.load(open(data_dir+'hamas_coord_users_lst.pkl','rb')) # keith's gt
with open(data_dir+'gaza_coordinated_accounts.txt','r') as f:
    coord_users = f.read().splitlines()
coord_users = set(coord_users)

df['label'] = df['twitterAuthorScreenname'].isin(coord_users).astype(int)

gt = df.groupby('twitterAuthorScreenname')['label'].first().loc[ordered_user_index,]
pickle.dump(gt.values,open(data_dir+'gt_data_luca.pkl','wb'))
logging.info(f"num regular users={len(gt[gt==0])}, num coord users={len(gt[gt==1])}")
logging.info('finished saving ground truth data')