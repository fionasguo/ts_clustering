import logging
import pandas as pd
import random
import numpy as np
import pickle
import os,gc
from datetime import datetime
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from glob import glob
import random
from preprocessing import preprocess_tweet
from collections import Counter


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

# matched_users = pd.read_csv('/nas/eclairnas01/users/siyiguo/reddit_hate_data/matched_users.csv')
# matched_users = matched_users[matched_users['category'].isin(['Xenophobic','Misogynistic','General hate'])]
# matched_users_dict = matched_users[['hate_user','random_user']].drop_duplicates(subset='hate_user').set_index('hate_user').to_dict()['random_user']

# hate_data_dir = '/nas/eclairnas01/users/siyiguo/hate_user_data'
# random_data_dir = '/nas/eclairnas01/users/siyiguo/reddit_hate_data/random_user_data'


# comments_files = glob(hate_data_dir+'/*_comments.json')
# random.shuffle(comments_files)
# df_comments = pd.DataFrame()
# df_submissions = pd.DataFrame()
# cnt = 0
# failed_cnt = 0
# failed_cnt_1 = 0
# failed_cnt_2 = 0
# failed_cnt_3 = 0
# for cf in comments_files:
#     hate_user_name = os.path.basename(cf).strip('_comments.json')
#     try:
#         rand_user_name = matched_users_dict[hate_user_name]
#         tmp_rand = pd.read_json(random_data_dir+'/'+rand_user_name+'_comments.json', lines=True)
#         tmp_rand = tmp_rand.drop_duplicates(subset=['created_utc','id'])
#         tmp_rand['hate_user'] = 0
#         df_comments = pd.concat([df_comments,tmp_rand])
#     except:
#         failed_cnt_1 += 1
#         continue
#     tmp_hate = pd.read_json(cf,lines=True)
#     tmp_hate = tmp_hate.drop_duplicates(subset=['created_utc','id'])
#     tmp_hate['hate_user'] = 1
#     df_comments = pd.concat([df_comments,tmp_hate])
    
#     if not os.path.exists(hate_data_dir+'/'+hate_user_name+'_submissions.json'):
#         failed_cnt += 1
#     else:
#         tmp_hate_submissions = pd.read_json(hate_data_dir+'/'+hate_user_name+'_submissions.json',lines=True)
#         tmp_hate_submissions = tmp_hate_submissions.drop_duplicates(subset=['created_utc','id'])
#         tmp_hate_submissions['hate_user'] = 1
#         df_submissions = pd.concat([df_submissions,tmp_hate_submissions])
        
#     if not os.path.exists(random_data_dir+'/'+rand_user_name+'_submissions.json'):
#         failed_cnt_3 += 1
#     else:
#         tmp_rand_submissions = pd.read_json(random_data_dir+'/'+rand_user_name+'_submissions.json', lines=True)
#         tmp_rand_submissions = tmp_rand_submissions.drop_duplicates(subset=['created_utc','id'])
#         tmp_rand_submissions['hate_user'] = 0
#         df_submissions = pd.concat([df_submissions,tmp_rand_submissions])

#     cnt += 1
#     if cnt % 1000 == 0:
#         print(cnt)
#     if cnt >= 10:
#         break
# df_comments = df_comments.drop_duplicates(subset=['created_utc','id'])
# df_comments.to_csv('/nas/eclairnas01/users/siyiguo/reddit_hate_data/10k_random_hate_users_comments.csv')
# print(f"{cnt} random and hate users_comments.csv SAVED, {df_comments.shape}\n{df_comments.columns}")

# df_submissions = df_submissions.drop_duplicates(subset=['created_utc','id'])
# df_submissions.to_csv('/nas/eclairnas01/users/siyiguo/reddit_hate_data/10k_random_hate_users_submissions.csv')
# print(f"{cnt} random and hate users_submissions.csv SAVED, {df_submissions.shape}\n{df_submissions.columns}")

# print(f"failed_count={failed_cnt}, failed_cnt_1={failed_cnt_1}, failed_cnt_2={failed_cnt_2}, failed_cnt_3={failed_cnt_3}")

##########################################################################################
####################### BERT embedding features ########################
# logging.info('start computing BERT embeddings')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size = 4096
# MODEL = 'sentence-transformers/stsb-xlm-r-multilingual' # 'cardiffnlp/twitter-xlm-roberta-base'
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModel.from_pretrained(MODEL).to(device)

# n_comp = 5

# data_dir = '/nas/eclairnas01/users/siyiguo/reddit_hate_data/'

# df = pd.read_csv(data_dir+'cleaned_reddit_hate_data.csv',lineterminator='\n')
# df['text'] = df['text'].astype(str).apply(preprocess_tweet)
# df['text'] = df['text'].str.replace('\n\n',' ').str.replace('\n',' ').str.replace('  ',' ')

# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     encoded_input = tokenizer(tmp['text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
# logging.info(f'BERT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'bert_embeddings.pkl','wb'))
# logging.info('BERT embeddings saved.')
# all_embeddings = pickle.load(open(data_dir+'bert_embeddings.pkl','rb'))
# logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')

# # dim reduction - pca
# logging.info('start PCA')
# all_embeddings = StandardScaler().fit_transform(all_embeddings)
# reducer = PCA(n_components=n_comp)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(data_dir+'bert_embeddings_pca.pkl','wb'))
# logging.info('PCA saved.')

#########################################################################################

# ###user_tot_tweet_threshold = 10

n_comp = 5

start_date = '2013-01-02'
end_date = '2022-12-31'
agg_time_period='3D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period) # ,tz='utc'

data_dir = '/nas/eclairnas01/users/siyiguo/reddit_hate_data/'

df = pd.read_csv(data_dir+'cleaned_reddit_hate_data.csv',lineterminator='\n')
df = df.reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])

all_embeddings = pickle.load(open(data_dir+'bert_embeddings_pca.pkl','rb'))
logging.info(f"loaded pca embeddings shape={all_embeddings.shape}")

df[list(range(n_comp))] = all_embeddings

user_ts_data = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='date')])[list(range(n_comp))].sum()
user_ts_data['tweet_count'] = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='date')])['id'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['author','date']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; len of entire time range: {len(entire_time_range)}')

# # transform into 3-d np array
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
# pickle.dump(ts_array[:,:,:-1], open(data_dir+'bert_embeddings_ts_data.pkl','wb'))
# logging.info('finished saving bert_embedding ts data')

# pickle.dump(ts_array[:,:,-1], open(data_dir+'activity_ts_data.pkl','wb'))
# logging.info('finished saving activity ts data')

# user_ts_data = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='date')])[list(range(n_comp))].mean()
# user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['author','date']),fill_value=0)
# ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# pickle.dump(ts_array, open(data_dir+'avg_embedding_ts_data.pkl','wb'))
# logging.info('finished saving avg embeddings ts data')

ordered_user_index = user_ts_data.groupby(level=0)[0].first().index

######################## avg user embedding across time ########################

# all_bert_embeddings = pickle.load(open(data_dir+'bert_embeddings.pkl','rb'))
# logging.info(f'loaded saved bert embeddings, shape: {all_bert_embeddings.shape}')

# avg_emb = np.empty((0,768))

# for user in tqdm(ordered_user_index):
#     avg_emb = np.vstack((avg_emb,all_bert_embeddings[df[df['author']==user].index.values,:].mean(axis=0)))

# pickle.dump(avg_emb,open(data_dir+'avg_user_bert_embeddings.pkl','wb'))
# logging.info(f'finished saving avg user bert embeddings averaged across time, shape: {avg_emb.shape}')

# ######################## demographic data ########################
# # build demographic data - 50 most posted subreddits

# df['subreddit'] = df['subreddit'].astype('category')
# df['subreddit_cat'] = df['subreddit'].cat.codes # int

# UNKNOWN_SUBREDDIT = df['subreddit_cat'].max()+1
# logging.info(f"number of subreddit categories: {df['subreddit_cat'].nunique()}")

# def top_subreddits(lst):
#     top_subr = [i[0] for i in Counter(lst).most_common(50) if i[0] != -1]
#     top_subr += [UNKNOWN_SUBREDDIT] * (50 - len(top_subr))  # pad with max+1, remove -1
#     return top_subr

# user_subreddits = df.groupby('author')['subreddit_cat'].apply(list) #.apply(len).value_counts()[:100]
# user_subreddits = user_subreddits.apply(top_subreddits)
# # user_subreddits.loc[user_subreddits['subreddit_cat']==-1,'subreddit_cat'] = df['subreddit_cat'].max()+1

# user_subreddits = np.array(user_subreddits.loc[ordered_user_index,].tolist())
# logging.info(f'demographic data - shape: {user_subreddits.shape}')
# pickle.dump(user_subreddits,open(data_dir+'demo_data.pkl','wb'))
# logging.info('finished saving demo data')


# ####################### ground truth data ########################
def fn2(x):
    modes = list(x.mode())
    if len(modes) > 1:
        return -1
    else:
        return modes[0]
    
# # gt_data = df.groupby('author')['hate_user'].apply(fn2)
# # gt_data = gt_data.loc[ordered_user_index,].values
# # logging.info(f'groundtruth data - shape: {gt_data.shape}, # hate users={gt_data.sum()}')
# # pickle.dump(gt_data,open(data_dir+'gt_data.pkl','wb'))

matched_users = pd.read_csv('/nas/eclairnas01/users/siyiguo/reddit_hate_data/cleaned_matched_users.csv')
matched_users_category = matched_users[['hate_user','category']].set_index('hate_user').to_dict()['category']

def fn3(row):
    if row['hate_user']==0:
        return 0
    return matched_users_category.get(row['author'],-1)

df['hate_user_category'] = df.apply(fn3,axis=1)
df.loc[df['hate_user_category']=='Misogynistic','hate_user_category']=1
df.loc[df['hate_user_category']=='Xenophobic','hate_user_category']=2
df.loc[df['hate_user_category']=='General hate','hate_user_category']=3

gt_categorical = df.groupby('author')['hate_user_category'].apply(lambda x: list(x.mode())[0])
gt_categorical = gt_categorical.loc[ordered_user_index,].values
pickle.dump(gt_categorical,open(data_dir+'gt_data_categorical.pkl','wb'))
logging.info(f'categorical groundtruth data - shape: {gt_categorical.shape}\n{Counter(gt_categorical)}')
