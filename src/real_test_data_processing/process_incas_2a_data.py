import logging
import pandas as pd
import random
import numpy as np
import pickle
import os,gc
from datetime import datetime
import logging
from tqdm import tqdm
from tweet_preprocessing import preprocess_tweet
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4096
MODEL = 'sentence-transformers/stsb-xlm-r-multilingual' # 'cardiffnlp/twitter-xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 10

user_tot_tweet_threshold = 3

start_date = '2023-01-01'
end_date = '2023-06-30'
agg_time_period='3D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period) # ,tz='utc'

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

data_dir = '/nas/eclairnas01/users/siyiguo/incas_2a_data/'

df = pd.read_csv(data_dir+'incas_2a_data.csv',lineterminator='\n')
logging.info(f'csv file loaded, shape={df.shape}')
df.loc[df['contentText'].isnull(),'contentText'] = ''
df['timePublished'] = pd.to_datetime(df.timePublished,unit='ms') #, unit='ms', utc=True
df = df[df['timePublished']<=pd.Timestamp('2023-06-30')]
logging.info(f"data shape={df.shape}, columns:\n{df.columns}")
logging.info(f"min date: {df['timePublished'].min()}, max date: {df['timePublished'].max()}")

user_tweet_count = df.groupby('twitterAuthorScreenname')['id'].count()
logging.info(f"total num of users: {len(user_tweet_count)}")
logging.info(f"num tweets per user: mean={user_tweet_count.mean()}, std={user_tweet_count.std()}")
logging.info(f"0.5 quantile={user_tweet_count.quantile(q=0.5)}, 0.75 quantile={user_tweet_count.quantile(q=0.75)}, 0.8 quantile={user_tweet_count.quantile(q=0.8)}, 0.9 quantile={user_tweet_count.quantile(q=0.9)}")

user_ts_count = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
logging.info('\n\n')

active_users = user_tweet_count[user_tweet_count>=user_tot_tweet_threshold]
active_user_set = set(active_users.index)

df = df[df['twitterAuthorScreenname'].isin(active_user_set)]
df = df.reset_index(drop=True)
logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/len(user_tweet_count)}')
logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')
user_ts_count = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
logging.info('\n\n')


####################### BERT embedding features ########################
# logging.info('start computing BERT embeddings')
# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     encoded_input = tokenizer(tmp['contentText'].apply(preprocess_tweet).tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
# logging.info(f'BERT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'xlmt_embeddings.pkl','wb'))
# logging.info('BERT embeddings saved.')

# # all_embeddings = pickle.load(open(data_dir+'bert_embeddings.pkl','rb'))
# # logging.info(f"bert embedding loaded shape={all_embeddings.shape}")

# # dim reduction - pca
# logging.info('start PCA')
# all_embeddings = StandardScaler().fit_transform(all_embeddings)
# reducer = PCA(n_components=n_comp)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+f'xlmt_embeddings_{n_comp}pca.pkl','wb'))
# logging.info(f'PCA saved. shape = {all_embeddings.shape}')

# # all_embeddings = pickle.load(open(data_dir+'bert_embeddings.pkl','rb'))
# # logging.info(f"loaded bert embeddings shape={all_embeddings.shape}")
# # df[list(range(768))] = all_embeddings
# # avg_user_embeddings = df.groupby('twitterAuthorScreenname')[list(range(768))].mean().loc[ordered_user_index,]
# # emb_array = np.array(avg_user_embeddings.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# # logging.info(f"emb array shape {emb_array.shape} mean {np.mean(emb_array,axis=1)}")
# # pickle.dump(emb_array,open(data_dir+'avg_user_bert_embeddings.pck','wb'))

all_embeddings = pickle.load(open(data_dir+f'xlmt_embeddings_{n_comp}pca.pkl','rb'))
logging.info(f"loaded pca embeddings shape={all_embeddings.shape}")

df[list(range(n_comp))] = all_embeddings


user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[list(range(n_comp))].sum()
user_ts_data['tweet_count'] = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
# pickle.dump(ts_array[:,:,:-1], open(data_dir+'xlmt_embeddings_ts_data.pkl','wb'))
# logging.info('finished saving xlmt_embedding ts data')

# pickle.dump(ts_array[:,:,-1], open(data_dir+'activity_ts_data.pkl','wb'))
# logging.info('finished saving activity ts data')

# # user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[list(range(n_comp))].mean()
# # user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
# # ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# # pickle.dump(ts_array, open(data_dir+'avg_embedding_ts_data.pkl','wb'))
# # logging.info('finished saving avg embeddings ts data')


ordered_user_index = user_ts_data.groupby(level=0)[0].first().index

######################## demographic data ########################
# # build demographic data
# demo_colnames = ['followerCount', 'followingCount']
# demo_data = df.groupby(['twitterAuthorScreenname'])[demo_colnames].sum()
# # make sure users are indexed in the same order

# demo_data = demo_data.loc[ordered_user_index,].values
# logging.info(f'demographic data - shape: {demo_data.shape}')
# pickle.dump(demo_data,open(data_dir+'demo_data.pkl','wb'))
# logging.info('finished saving demo data')

####################### ground truth data ########################
# get ground truth data
with open(data_dir+'incas_2a_coord_users_keith.txt','r') as f:
    coord_users = f.read().splitlines()
coord_users = set(coord_users)

df['label'] = df['twitterAuthorScreenname'].isin(coord_users).astype(int)

gt = df.groupby('twitterAuthorScreenname')['label'].first().loc[ordered_user_index,]
pickle.dump(gt.values,open(data_dir+'gt_data.pkl','wb'))
logging.info(f"num regular users={len(gt[gt==0])}, num coord users={len(gt[gt==1])}")
logging.info('finished saving ground truth data')


# ######################## retweet links data ########################

df['retweeted_user_id'] = np.NaN
df.loc[df['engagementType']=='retweet','retweeted_user_id'] = df.loc[df['engagementType']=='retweet','engagementParentId']

def fn_a(lst):
    return [dic[x] for x in lst if dic.get(x) is not None]

user_links = df.groupby(['twitterAuthorScreenname'])['retweeted_user_id'].apply(list)
user_links = pd.DataFrame(user_links)
user_links = user_links.loc[ordered_user_index,]
dic = {u:i for i,u in enumerate(list(user_links.index))}
user_links['retweeted_user_id'] = user_links['retweeted_user_id'].apply(fn_a)
max_links_len = user_links['retweeted_user_id'].apply(len).max()
logging.info(f"user links: max_links_len={max_links_len}, avg links len = {user_links['retweeted_user_id'].apply(len).mean()}, median links len={user_links['retweeted_user_id'].apply(len).median()}")
user_links_array = user_links['retweeted_user_id'].tolist()
user_links_array = np.array([(n+[-1]*max_links_len)[:max_links_len] for n in user_links_array])
logging.info(f'shape of np array for the user links data: {user_links_array.shape}')
pickle.dump(user_links_array, open(data_dir+'links_data.pkl','wb'))
