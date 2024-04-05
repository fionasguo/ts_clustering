import pandas as pd
import numpy as np
import re
import gzip
import gc
from os import listdir
from os.path import isfile, join, isdir
from glob import glob
import pickle
import os
from datetime import datetime
import logging
from tweet_preprocessing import preprocess_tweet
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size = 2048
# MODEL = 'sentence-transformers/stsb-xlm-r-multilingual' # 'cardiffnlp/twitter-xlm-roberta-base'
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 5

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

# # combine control and drivers together for each country
# data_dir = '/nas/eclairnas01/users/siyiguo/hashed_infoOps/'
# nations = ['cnhu','venezuela','egyptuae']

# for n in nations:
#     path = data_dir+n+'/'
#     infoOps = [f for f in listdir(path)]
#     logging.info(f"Nation: {n}, num of files: {len(infoOps)}")

#     control = pd.DataFrame()
#     treated = pd.DataFrame()
#     for file in infoOps:
#         if 'control_user' in file:
#             control = pd.concat([control, pd.read_csv(path+file, lineterminator='\n')])
#             logging.info(f"control file {file}, shape: {control.shape}\n{control.columns}\n")
#         else:
#             treated = pd.concat([treated, pd.read_csv(path+file, lineterminator='\n')])
#             logging.info(f"treated file {file}, shape: {treated.shape}\n{treated.columns}\n")
#     control['label'] = 0
#     treated['label'] = 1

#     # process control data
#     if 'userid' not in list(control.columns):
#         tmp = control['user'].apply(pd.Series)
#         tmp = tmp.rename(columns={k:'user_'+k for k in list(tmp.columns)})
#         control = pd.concat([control.drop(['user'], axis=1), tmp], axis=1)
#     control = control.rename(columns={'id':'tweetid','full_text':'tweet_text','lang':'tweet_language','created_at':'tweet_time',
#                                       'user_id':'userid','user_description':'user_profile_description',
#                                       'user_followers_count':'follower_count','user_friends_count':'following_count',
#                                       'user_location':'user_reported_location'})

#     logging.info(f"agg control file: shape={control.shape}, min date:{control['tweet_time'].min()}, max date:{control['tweet_time'].max()}, num users: {len(pd.unique(control['userid']))}\n{control.columns}")
#     logging.info(f"agg treated file: shape={treated.shape}, min date:{treated['tweet_time'].min()}, max date:{treated['tweet_time'].max()}, num users: {len(pd.unique(treated['userid']))}\n{treated.columns}")

#     target_cols = ['tweetid', 'userid', 'user_screen_name', 'user_profile_description',
#        'follower_count', 'following_count','tweet_language',
#        'tweet_text', 'tweet_time', 'label']
#     control = control[target_cols]
#     treated = treated[target_cols]

#     pd.concat([control,treated],axis=0).to_csv(data_dir+n+'/all_controls_drivers.csv',index=False)
#     logging.info(f"{n} saved. control shape={control.shape}, treated shape={treated.shape}")


"""
control.columns
Index(['id', 'full_text', 'source', 'lang', 'user', 'geo', 'coordinates',
       'place', 'in_reply_to_status_id', 'in_reply_to_screen_name',
       'in_reply_to_user_id', 'created_at', 'entities'],
      dtype='object')

control.loc[0,'user'].keys()
dict_keys(['id', 'url', 'location', 'description', 'verified', 'public_metrics', 'profile_image_url', 'entities', 'protected', 'created_at', 'name', 'screen_name', 'followers_count', 'friends_count', 'statuses_count', 'listed_count', 'favourites_count'])


treated.columns
Index(['tweetid', 'userid', 'user_display_name', 'user_screen_name',
       'user_reported_location', 'user_profile_description',
       'user_profile_url', 'follower_count', 'following_count',
       'account_creation_date', 'account_language', 'tweet_language',
       'tweet_text', 'tweet_time', 'tweet_client_name', 'in_reply_to_userid',
       'in_reply_to_tweetid', 'quoted_tweet_tweetid', 'is_retweet',
       'retweet_userid', 'retweet_tweetid', 'latitude', 'longitude',
       'quote_count', 'reply_count', 'like_count', 'retweet_count', 'hashtags',
       'urls', 'user_mentions', 'poll_choices'],
      dtype='object')
"""


"""
egyptuae: '2016-01-01' to '2019-08-05', 3D
venezuela: '2017-01-01' to '2021-06-03' 3D
cnhu: '2019-04-22' to '2021-04-04' 3D
"""


start_date = '2019-04-21'
end_date = '2021-04-05'
agg_time_period='3D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period,tz='utc')

data_dir = '/nas/eclairnas01/users/siyiguo/hashed_infoOps/cnhu/'

df = pd.read_csv(data_dir+'all_controls_drivers.csv',lineterminator='\n')

df['tweet_time'] = pd.to_datetime(df['tweet_time'],utc=True,format='mixed')
logging.info(f"min date:{df['tweet_time'].min()}, max date:{df['tweet_time'].max()}")
logging.info(f"languages:\n{df['tweet_language'].value_counts()}")
logging.info(f"time distribution:\n{df['tweet_time'].dt.year.value_counts()}")
logging.info(f"time zone check: {df['tweet_time'].apply(lambda t: t.tzinfo is None).sum()}, {pd.unique(df['tweet_time'].apply(lambda t: t.tzinfo))}")

df = df[(df['tweet_time']>=pd.Timestamp(start_date,tz='utc')) & (df['tweet_time']<=pd.Timestamp(end_date,tz='utc'))]
logging.info(f"data {start_date} to {end_date} {agg_time_period}: shape={df.shape}")

user_tweet_count = df.groupby('userid')['tweetid'].count()
logging.info(f"total num of users: {len(user_tweet_count)}")
logging.info(f"num tweets per user: mean={user_tweet_count.mean()}, std={user_tweet_count.std()}")
logging.info(f"0.5 quantile={user_tweet_count.quantile(q=0.5)}, 0.75 quantile={user_tweet_count.quantile(q=0.75)}, 0.8 quantile={user_tweet_count.quantile(q=0.8)}, 0.9 quantile={user_tweet_count.quantile(q=0.9)}")

user_ts_count = df.groupby(['userid',pd.Grouper(freq=agg_time_period,key='tweet_time')])['tweetid'].count()
logging.info(user_ts_count)
user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['userid','tweet_time']),fill_value=0)
logging.info(user_ts_count_)
logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
logging.info('\n\n')

######################## BERT embedding features ########################
# logging.info('start computing XLMT embeddings')
# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     tmp['tweet_text'] = tmp['tweet_text'].apply(preprocess_tweet)
#     encoded_input = tokenizer(tmp['tweet_text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
#     if all_embeddings.shape[0] >= 5000000:
#         logging.info(f'saving intermediate XLMT embeddings at batch={i}, embeddings shape: {all_embeddings.shape}')
#         pickle.dump(all_embeddings,open(f'{data_dir}xlmt_embeddings_{i}.pkl','wb'))
#         all_embeddings = np.empty((0,768))

# logging.info(f'XLMT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(f'{data_dir}xlmt_embeddings_{i}.pkl','wb'))
# logging.info('XLMT embeddings saved.')

# del all_embeddings
# del df
# gc.collect()

# all_embeddings = np.empty((0,768))
# emb_files = glob(data_dir+'xlmt_embeddings_*.pkl')
# for f in emb_files:
#     logging.info(f)
#     tmp = pickle.load(open(f,'rb'))
#     idx = np.random.randint(len(tmp), size=len(tmp)//2)
#     tmp = tmp[idx,:]
#     all_embeddings = np.vstack((all_embeddings,tmp))

#     del tmp
#     gc.collect()
# logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')
# # pickle.dump(all_embeddings,open(f'{data_dir}xlmt_embeddings.pkl','wb'))
# # logging.info(f"all embeddings saved")

# # dim reduction - pca
# logging.info('start PCA')
# # OOM - sample 50% data to train PCA and transform the rest
# # idx = np.random.randint(len(all_embeddings), size=len(all_embeddings)//2)
# # all_embeddings = all_embeddings[idx,:]
# std_scaler = StandardScaler()
# std_scaler = std_scaler.fit(all_embeddings)
# all_embeddings = std_scaler.fit_transform(all_embeddings)
# pickle.dump(std_scaler,open(data_dir+'std_scaler_model.pkl','wb'))
# logging.info("standardized embeddings")

# reducer = PCA(n_components=n_comp)
# reducer = reducer.fit(all_embeddings)
# pickle.dump(reducer,open(data_dir+'pca_model.pkl','wb'))
# logging.info('PCA model saved')

# std_scaler = pickle.load(open(data_dir+'std_scaler_model.pkl','rb'))
# reducer = pickle.load(open(data_dir+'pca_model.pkl','rb'))

# emb_files = glob(data_dir+'xlmt_embeddings_*.pkl')
# for f in emb_files:
#     # logging.info(pickle.load(open(f,'rb')).shape)
#     all_embeddings = pickle.load(open(f,'rb'))
#     all_embeddings = std_scaler.transform(all_embeddings)
#     all_embeddings = reducer.transform(all_embeddings)
#     pickle.dump(all_embeddings,open(f"{f.split('.')[0]}_pca.pkl",'wb'))
#     logging.info(f'saved bert embeddings pca, shape: {all_embeddings.shape}')

# del all_embeddings
# del reducer
# del std_scaler
# gc.collect()

# all_embeddings = np.empty((0,5))
# pca_files =  glob(data_dir+'xlmt_embeddings_*_pca.pkl')
# for f in pca_files:
#     all_embeddings = np.vstack((all_embeddings,pickle.load(open(f,'rb'))))

# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'xlmt_embeddings_pca.pkl','wb'))
# logging.info("PCA embeddings saved.")

# ts data with another set of features - bert embedding - umap reduced
# logging.info('start computing XLMT embeddings')
# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     encoded_input = tokenizer(tmp['tweet_text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
# logging.info(f'XLMT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'xlmt_embeddings.pkl','wb'))
# logging.info('XLMT embeddings saved.')
# all_embeddings = pickle.load(open(data_dir+'xlmt_embeddings.pkl','rb'))
# logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')

# del df
# gc.collect()

# # all_embeddings = np.empty((0,768))
# # emb_files = glob(data_dir+'xlmt_embeddings_*.pkl')
# # for f in emb_files:
# #     # logging.info(pickle.load(open(f,'rb')).shape)
# #     all_embeddings = np.vstack((all_embeddings,pickle.load(open(f,'rb'))))
# # logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')
# # idx = np.random.randint(len(all_embeddings), size=len(all_embeddings)//2)
# # all_embeddings = all_embeddings[idx,:]
# # std_scaler = StandardScaler()
# # std_scaler = std_scaler.fit(all_embeddings)
# # all_embeddings = std_scaler.fit_transform(all_embeddings)
# # pickle.dump(std_scaler,open(data_dir+'std_scaler_model.pkl','wb'))
# # logging.info("standardized embeddings")

# # reducer = PCA(n_components=n_comp)
# # reducer = reducer.fit(all_embeddings)
# # pickle.dump(reducer,open(data_dir+'pca_model.pkl','wb'))
# # logging.info('PCA model saved')

# # std_scaler = pickle.load(open(data_dir+'std_scaler_model.pkl','rb'))
# # reducer = pickle.load(open(data_dir+'pca_model.pkl','rb'))

# # emb_files = glob(data_dir+'xlmt_embeddings_*.pkl')
# # for f in emb_files:
# #     # logging.info(pickle.load(open(f,'rb')).shape)
# #     all_embeddings = pickle.load(open(f,'rb'))
# #     all_embeddings = std_scaler.transform(all_embeddings)
# #     all_embeddings = reducer.transform(all_embeddings)
# #     pickle.dump(all_embeddings,open(f"{f.split('.')[0]}_pca.pkl",'wb'))
# #     logging.info(f'saved bert embeddings pca, shape: {all_embeddings.shape}')

# # # dim reduction - UMAP - OOM
# # reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',verbose=True)
# # all_embeddings = reducer.fit_transform(all_embeddings)
# # logging.info(f'UMAP finshed, dimension reduced embeddings shape: {all_embeddings.shape}')

# all_embeddings = pickle.load(open(f'{data_dir}xlmt_embeddings.pkl','rb'))
# dim reduction - pca
# logging.info('start PCA')
# all_embeddings = StandardScaler().fit_transform(all_embeddings)
# reducer = PCA(n_components=n_comp)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'xlmt_embeddings_pca.pkl','wb'))
# logging.info('PCA saved')

# # all_embeddings = np.empty((0,5))
# # pca_files =  glob(data_dir+'xlmt_embeddings_*_pca.pkl')
# # for f in pca_files:
# #     all_embeddings = np.vstack((all_embeddings,pickle.load(open(f,'rb'))))

# # logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# # pickle.dump(all_embeddings,open(data_dir+'xlmt_embeddings_pca.pkl','wb'))
# # logging.info("PCA embeddings saved.")

all_embeddings = pickle.load(open(data_dir+'xlmt_embeddings_pca.pkl','rb'))
logging.info("PCA embeddings loaded.")

df[list(range(n_comp))] = all_embeddings

start_date = '2019-04-22'
end_date = '2021-04-04'
agg_time_period='3D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period,tz='utc')
df = df[(df['tweet_time']>=pd.Timestamp(start_date,tz='utc')) & (df['tweet_time']<=pd.Timestamp(end_date,tz='utc'))]
logging.info(f"data '2019-04-22' to '2021-04-04' 3D: shape={df.shape}")

user_ts_data = df.groupby(['userid',pd.Grouper(freq=agg_time_period,key='tweet_time')])[list(range(n_comp))].sum()
user_ts_data['tweet_count'] = df.groupby(['userid',pd.Grouper(freq=agg_time_period,key='tweet_time')])['tweetid'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['userid','tweet_time']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}')

# transform into 3-d np array
ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
pickle.dump(ts_array[:,:,:5], open(data_dir+'xlmt_embeddings_ts_data.pkl','wb'))
logging.info('finished saving xlmt_embedding ts data')
pickle.dump(ts_array[:,:,-1], open(data_dir+'activity_ts_data.pkl','wb'))
logging.info('finished saving activity ts data')

# keep record to make sure users are indexed in the same order
ordered_user_index = user_ts_data.groupby(level=0)[0].first().index


# ######################## demographic data ########################
# def fn1(x):
#     modes = list(x.mode())
#     if len(modes) > 1:
#         return 'und'
#     else:
#         return modes[0]

demo_colnames = ['follower_count', 'following_count']
# build demographic data
demo_data = df.groupby(['userid'])[demo_colnames].first()
# # process language as lookup table
# logging.info(f"processing language categories to int: total num of lang={len(pd.unique(demo_data['account_language']))}\n{pd.unique(demo_data['account_language'])}")
# tmp = pd.Categorical(demo_data['account_language'])
# demo_data['account_language'] = tmp.codes
# demo_data.loc[demo_data['account_language']==-1,'account_language'] = 3
# tmp = df.groupby(['userid'])['tweet_language'].apply(fn1)
# tmp = pd.Categorical(tmp)
# demo_data['tweet_language'] = tmp.codes
# logging.info(f"users' most freq tweet lanuguage {tmp.categories}")
# logging.info(f"after lang to code:  total num of lang={len(pd.unique(demo_data['account_language']))}\n{pd.unique(demo_data['account_language'])}")
# save
demo_data = demo_data.loc[ordered_user_index,].values
logging.info(f'demographic data - shape: {demo_data.shape}')
pickle.dump(demo_data,open(data_dir+'demo_data.pkl','wb'))

# """
# target_cols = ['tweetid', 'userid', 'user_screen_name',
# #        'user_reported_location', 'user_profile_description',
# #        'follower_count', 'following_count','tweet_language',
# #        'tweet_text', 'tweet_time', 'label']
# """
# ######################## ground truth data ########################
def fn2(x):
    modes = list(x.mode())
    if len(modes) > 1:
        return 1
    else:
        return modes[0]
    
gt_data = df.groupby('userid')['label'].apply(fn2)
gt_data = gt_data.loc[ordered_user_index,].values
logging.info(f'groundtruth data - shape: {gt_data.shape}, # coord={gt_data.sum()}')
pickle.dump(gt_data,open(data_dir+'/gt_data.pkl','wb'))