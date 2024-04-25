import os
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

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
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True

create_logger()

data_dir = '/nas/eclairnas01/users/siyiguo/eating_disorder_data/df_emo_RT_toxicity_scores.csv'

user_tot_tweet_threshold = 5
start_date = '2022-10-24'
end_date = '2023-03-10'
agg_time_period='2D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period,tz='utc')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2048
MODEL = 'vinai/bertweet-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 5

df = pd.read_csv(data_dir,lineterminator='\n')
logging.info(f"data loaded, shape: {df.shape}, columns:")
logging.info(df.columns)

df['created_at'] = pd.to_datetime(df['created_at']) # utc
logging.info(f"min date:{df['created_at'].min()}, max date:{df['created_at'].max()}")

user_tweet_count = df.groupby('author_id')['tweet_id'].count()
logging.info(f"total num of users: {len(user_tweet_count)}")
logging.info(f"num tweets per user: mean={user_tweet_count.mean()}, std={user_tweet_count.std()}")
logging.info(f"0.5 quantile={user_tweet_count.quantile(q=0.5)}, 0.75 quantile={user_tweet_count.quantile(q=0.75)}, 0.8 quantile={user_tweet_count.quantile(q=0.8)}, 0.9 quantile={user_tweet_count.quantile(q=0.9)}")

user_ts_count = df.groupby(['author_id',pd.Grouper(freq=agg_time_period,key='created_at')])['tweet_id'].count()
user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['author_id','created_at']),fill_value=0)
logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")

active_users = user_tweet_count[user_tweet_count>=user_tot_tweet_threshold]
active_user_set = set(active_users.index)
df = df[df['author_id'].isin(active_user_set)]
df = df.reset_index(drop=True)
logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/len(user_tweet_count)}')
logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')

######################## BERT embedding features ########################
# ts data with another set of features - bert embedding - umap reduced
logging.info('start computing BERT embeddings')
# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     encoded_input = tokenizer(tmp['text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
# logging.info(f'BERT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/bert_embeddings.pkl','wb'))
# logging.info('BERT embeddings saved.')
all_embeddings = pickle.load(open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/bert_embeddings.pkl','rb'))
logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')

# # dim reduction - UMAP - OOM
# reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',verbose=True)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'UMAP finshed, dimension reduced embeddings shape: {all_embeddings.shape}')

# dim reduction - pca
logging.info('start PCA')
all_embeddings = StandardScaler().fit_transform(all_embeddings)
reducer = PCA(n_components=n_comp)
all_embeddings = reducer.fit_transform(all_embeddings)
logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
df[list(range(n_comp))] = all_embeddings

df = df[df['community_id']<=6]
logging.info(f"first 7 communities, shape={df.shape}")

user_ts_data = df.groupby(['author_id',pd.Grouper(freq=agg_time_period,key='created_at')])[list(range(n_comp))].sum()
user_ts_data['tweet_count'] = df.groupby(['author_id',pd.Grouper(freq=agg_time_period,key='created_at')])['tweet_id'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['author_id','created_at']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(active_user_set)}, len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
# pickle.dump(ts_array, open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/bert_embeddings_ts_data.pkl','wb'))
# logging.info('finished saving BERT embeddings ts data')


######################## ground truth data - hashtag_coord ########################
# get ground truth data
ordered_user_index = user_ts_data.groupby(level=0)['tweet_count'].first().index

gt = df.groupby('author_id')['community_id'].first().loc[ordered_user_index,]
pickle.dump(gt.values,open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/gt_data.pkl','wb'))
logging.info('finished saving ground truth data')

df['neg_emot'] = df['Anger'] + df['Disgust']
df['pos_emot'] = df['Joy'] + df['Optimism'] + df['Love']
df['total_toxicity'] = df['toxicity'] + df['severe_toxicity'] + df['obscene'] + df['threat'] + df['insult'] + df['identity_attack']
gt_neg = df.groupby('author_id')['neg_emot'].mean().loc[ordered_user_index,]
pickle.dump(gt_neg.values,open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/gt_data_neg_emot.pkl','wb'))
gt_pos = df.groupby('author_id')['pos_emot'].mean().loc[ordered_user_index,]
pickle.dump(gt_pos.values,open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/gt_data_pos_emot.pkl','wb'))
gt_toxicity = df.groupby('author_id')['total_toxicity'].mean().loc[ordered_user_index,]
pickle.dump(gt_toxicity.values,open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/gt_data_toxicity.pkl','wb'))


# ######################## retweet links data ########################
# def fn_a(lst):
#     return [dic[x] for x in lst if dic.get(x) is not None]

# user_links = df.groupby(['author_id'])['parent_author_id'].apply(list)
# user_links = user_links.loc[ordered_user_index,]
# dic = {u:i for i,u in enumerate(list(user_links.index))}
# user_links = user_links.apply(fn_a)
# max_links_len = user_links.apply(len).max()
# logging.info(f"user links: max_links_len={max_links_len}, avg links len = {user_links.apply(len).mean()}, median links len={user_links.apply(len).median()}")
# user_links_array = user_links.tolist()
# user_links_array = np.array([(n+[-1]*max_links_len)[:max_links_len] for n in user_links_array])
# logging.info(f'shape of np array for the user links data: {user_links_array.shape}')
# pickle.dump(user_links_array, open('/nas/eclairnas01/users/siyiguo/eating_disorder_data/links_data.pkl','wb'))