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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4096
MODEL = 'sentence-transformers/stsb-xlm-r-multilingual' # 'cardiffnlp/twitter-xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 20

user_tot_tweet_threshold = 10

start_date = '2023-08-01'
end_date = '2023-11-30'
agg_time_period='2D'
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

data_dir = '/nas/eclairnas01/users/siyiguo/hamas_data/reddit/'

# df = pd.read_csv(data_dir+'hammas_reddit_comments_submissions_active_users.csv',lineterminator='\n')
# logging.info(f'csv file loaded, shape={df.shape}')
# df.loc[df['text'].isnull(),'text'] = ''
# df['timestamp'] = pd.to_datetime(df.timestamp) #, unit='ms', utc=True
# logging.info(f"data shape={df.shape}, columns:\n{df.columns}")
# logging.info(f"min date: {df['timestamp'].min()}, max date: {df['timestamp'].max()}")

# # user_tweet_count = df.groupby('author')['id'].count()
# # logging.info(f"total num of users: {len(user_tweet_count)}")
# # logging.info(f"num tweets per user: mean={user_tweet_count.mean()}, std={user_tweet_count.std()}")
# # logging.info(f"0.5 quantile={user_tweet_count.quantile(q=0.5)}, 0.75 quantile={user_tweet_count.quantile(q=0.75)}, 0.8 quantile={user_tweet_count.quantile(q=0.8)}, 0.9 quantile={user_tweet_count.quantile(q=0.9)}")

# # user_ts_count = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='timestamp')])['id'].count()
# # user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['author','timestamp']),fill_value=0)
# # logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
# # logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
# # logging.info('\n\n')

# # # active_users = user_tweet_count[user_tweet_count>=user_tot_tweet_threshold]
# # # active_user_set = set(active_users.index)

# # # df = df[df['author'].isin(active_user_set)]
# # # df = df.reset_index(drop=True)
# # # logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/len(user_tweet_count)}')
# # # logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')
# # # user_ts_count = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='timestamp')])['id'].count()
# # # user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['author','timestamp']),fill_value=0)
# # # logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
# # # logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
# # # logging.info('\n\n')

# # # authors_w_10_texts = df.groupby('author')['text'].apply(list).apply(lambda x: [i for i in x if len(i.split())>4]).apply(lambda x: random.sample(x,k=min(10,len(x))))
# # # authors_w_10_texts = authors_w_10_texts[authors_w_10_texts[authors_w_10_texts.apply(len)>=3]]
# # # active_user_set = set(authors_w_10_texts.index)

# # # df = df[df['author'].isin(active_user_set)]
# # # df = df.reset_index(drop=True)
# # # logging.info(f'data with authors having more than 3 texts longer than 4 words - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/len(user_tweet_count)}')
# # # logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')
# # # user_ts_count = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='timestamp')])['id'].count()
# # # user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['author','timestamp']),fill_value=0)
# # # logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
# # # logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")
# # # logging.info('\n\n')


# # """
# # df.columns
# # Index(['subreddit', 'id', 'text', 'author', 'timestamp', 'submission_id',
# #        'controversial', 'score', 'ups', 'downs', 'parent_id', 'type',
# #        'content', 'upvote_ratio'],
# #       dtype='object')

# # """

# # ######################## BERT embedding features ########################
# # # logging.info('start computing BERT embeddings')
# # # all_embeddings = np.empty((0,768))
# # # for i in tqdm(range(len(df)//batch_size+1)):
# # #     tmp = df[i*batch_size:(i+1)*batch_size]
# # #     encoded_input = tokenizer(tmp['text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
# # #     with torch.no_grad():
# # #         embeddings = model(**encoded_input).pooler_output
# # #     embeddings = embeddings.cpu().numpy()
# # #     all_embeddings = np.vstack((all_embeddings,embeddings))
# # # logging.info(f'BERT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# # # pickle.dump(all_embeddings,open(data_dir+'bert_embeddings.pkl','wb'))
# # # logging.info('BERT embeddings saved.')
# # # # all_embeddings = pickle.load(open('/nas/eclairnas01/users/siyiguo/incas_data/phase1a_bert_embeddings.pkl','rb'))
# # # # logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')

# # # # dim reduction - pca
# # # logging.info('start PCA')
# # # all_embeddings = StandardScaler().fit_transform(all_embeddings)
# # # reducer = PCA(n_components=n_comp)
# # # all_embeddings = reducer.fit_transform(all_embeddings)
# # # logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# # # pickle.dump(all_embeddings,open(data_dir+'bert_embeddings_pca.pkl','wb'))
# # # logging.info('PCA saved.')

# # gt = pd.read_csv(data_dir+'hamas_reddit_gt_data.csv',lineterminator='\n')
# # gt = gt.reset_index(drop=True)[(~gt['Israel'].isnull()) & (~gt['Palestine'].isnull())]
# # logging.info(f"gt shape {gt.shape}")
# # gt = gt.set_index('author')
# # ordered_user_index = gt.index

# # all_embeddings = pickle.load(open(data_dir+'bert_embeddings.pkl','rb'))
# # logging.info(f"loaded bert embeddings shape={all_embeddings.shape}")
# # df[list(range(768))] = all_embeddings
# # avg_user_embeddings = df.groupby('author')[list(range(768))].mean().loc[ordered_user_index,]
# # emb_array = np.array(avg_user_embeddings.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# # logging.info(f"emb array shape {emb_array.shape} mean {np.mean(emb_array,axis=1)}")
# # pickle.dump(emb_array,open(data_dir+'avg_user_bert_embeddings.pck','wb'))

# # all_embeddings = pickle.load(open(data_dir+'bert_embeddings_pca.pkl','rb'))
# # logging.info(f"loaded pca embeddings shape={all_embeddings.shape}")

# # df[list(range(n_comp))] = all_embeddings


# # target_authors = set(['98evpe98', 'AmbzbLfu', 'Aojdl3577', 'BSbtppm', 'Bddpnqmjtife-Bsu6245',
# #        'Bddpvoubc2mjuz', 'BnJOpuBmqibsjvt', 'BnbajohQJOHBT', 'Bvup_Qjf',
# #        'Cvttgpsuph70', 'DYA226', 'Dmpxocvdl', 'EfmjdjpvtXbs6482',
# #        'Epou_upvdi_nz_spdl', 'Espof_Spo', 'Fbtufso-Bqsjdpu-903',
# #        'FmjabcfuiTqbhifuuj', 'Fobcmjoh_Uvsumf', 'GEJDifn', 'GfuuzCppgCpu',
# #        'Gmpsjqb06', 'Hbmjnlbmjn', 'HsjnnKbdlfsKbdl', 'IBTTBO-fmkf23',
# #        'IbhhjtQpqf', 'IfbwzQptu2', 'JTipvmeCXpsljo', 'Lsov888',
# #        'Mjgf-Qjduvsf7430', 'MvdbtPJoupyjdbep', 'MvsljohHvz', 'MzbTubsl',
# #        'Nbspljjj', 'Nbtufs_Dsz_6006', 'Njtufs_Tqbdfmz', 'Npop2924',
# #        'NzObnfJtTjnpo99', 'Pckfdujwf_Ipssps2600', 'Qbmf-Xpsmemjoftt8118',
# #        'Qiboupn-Qboeb3329', 'SfbmCsboeOfx', 'SfbmjtujdUbeqpmf2037',
# #        'Spojo_Z3L', 'SpvoeBe4979', 'TOQ36', 'Tfsjpvt-Gmpx-', 'Tivsh',
# #        'Tpepofxjuimjggfs', 'TqbdfCpxjf3119', 'Tqbdf_Dpx-cpz', 'TufxjfTXT',
# #        'Tupsz_5_fwfszuijoh', 'TxfOboj', 'UibuGmzjohTdputnbo', 'Uif-Ovjtbodf',
# #        '_AjjppjjA_', 'ebwabs0', 'epobmejopp', 'exbjs', 'fhhtbmbetboexjdijtn',
# #        'gsjfoemzdmpdl313', 'hmjuu4scvooj', 'ibmmboebmf', 'ibqqzhbup',
# #        'ijijij484', 'kfggxvmg', 'ktltltl6', 'luljuujft', 'njoenpvoubjo',
# #        'obnfjtublfo-3', 'op_uibol_v_qma', 'pmelbsm3399', 'svqfsubmefstpo',
# #        'tipdlfsezfsnpn', 'tjhbwfo', 'xbjubnjovufxifsfjbn', 'xftu1of',
# #        'xjuiuifjotjefpvu', 'yxpset60', 'zpvejeousfeeju'])
# # df = df[df['authors'].isin(target_authors)]


# user_ts_data = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='timestamp')])[['ups','downs']].sum()
# user_ts_data['tweet_count'] = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='timestamp')])['id'].count()
# logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# # fill the time series with the entire time range
# user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['author','timestamp']),fill_value=0)
# logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; len of entire time range: {len(entire_time_range)}')

# # # transform into 3-d np array
# # ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# # logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
# # pickle.dump(ts_array[:,:,:-1], open(data_dir+'xlmt_embeddings_ts_data.pkl','wb'))
# # logging.info('finished saving xlmt_embedding ts data')

# # pickle.dump(ts_array[:,:,-1], open(data_dir+'activity_ts_data.pkl','wb'))
# # logging.info('finished saving activity ts data')

# # user_ts_data = df.groupby(['author',pd.Grouper(freq=agg_time_period,key='timestamp')])[list(range(n_comp))].mean()
# # user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['author','timestamp']),fill_value=0)
# # ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
# # pickle.dump(ts_array, open(data_dir+'avg_embedding_ts_data.pkl','wb'))
# # logging.info('finished saving avg embeddings ts data')


# ordered_user_index = user_ts_data.groupby(level=0)['ups'].first().index

# # ######################## demographic data ########################
# # # build demographic data
# # demo_colnames = ['ups', 'downs']
# # demo_data = df.groupby(['author'])[demo_colnames].sum()
# # # make sure users are indexed in the same order

# # demo_data = demo_data.loc[ordered_user_index,].values
# # logging.info(f'demographic data - shape: {demo_data.shape}')
# # pickle.dump(demo_data,open(data_dir+'demo_data.pkl','wb'))
# # logging.info('finished saving demo data')

# # # ####################### ground truth data ########################
# # # # get ground truth data
# # # with open(data_dir+'gaza_coordinated_accounts.txt','r') as f:
# # #     coord_users = f.read().splitlines()
# # # coord_users = set(coord_users)

# # # df['label'] = df['author'].isin(coord_users).astype(int)

# # # gt = df.groupby('author')['label'].first().loc[ordered_user_index,]
# # # pickle.dump(gt.values,open(data_dir+'gt_data_luca.pkl','wb'))
# # # logging.info(f"num regular users={len(gt[gt==0])}, num coord users={len(gt[gt==1])}")
# # # logging.info('finished saving ground truth data')

# # # authors_w_10_texts.loc[ordered_user_index].to_csv(data_dir+'authors_w_10_texts.csv')

# # ######################## retweet links data ########################

# gt = pd.read_csv(data_dir+'hamas_reddit_gt_data.csv',lineterminator='\n')
# gt = gt.reset_index(drop=True)[(~gt['Israel'].isnull()) & (~gt['Palestine'].isnull())]
# logging.info(f"gt shape {gt.shape}")
# gt = gt.set_index('author')
# ordered_user_index = gt.index

# subreddit_dic = df.groupby('subreddit')['author'].apply(list).to_dict()
# conflict_keys = [
#     'israelexposed','exmuslim', 'Jewish', 'Judaism', 'IsraelCrimes', 'Palestinian_Violence', 'AntiSemitismInReddit',
#     'IsraelUnderAttack', 'IsraelICYMI', 'MuslimLounge', 'Muslim', 'MuslimCorner']
# conf_subreddit_dic = {key: subreddit_dic[key] for key in conflict_keys}

# dic = {u:i for i,u in enumerate(list(ordered_user_index))}

# def fn_b(lst):
#     result = []
#     for l in lst:
#         if conf_subreddit_dic.get(l):
#             result.extend(conf_subreddit_dic.get(l))
    
#     return list(set([dic[r] for r in result if dic.get(r) is not None]))

# tqdm.pandas()

# user_links = df.groupby('author')['subreddit'].apply(lambda x: list(set(x))).progress_apply(fn_b)
# user_links = user_links.loc[ordered_user_index,]
# max_links_len = user_links.apply(len).max()
# logging.info(f"user links: max_links_len={max_links_len}, avg links len = {user_links.apply(len).mean()}, std links len={user_links.apply(len).std()}")

# del df
# gc.collect()

# user_links_array = user_links.tolist()
# user_links_array = np.array([(n+[-1]*max_links_len)[:max_links_len] for n in user_links_array])
# logging.info(f'shape of np array for the user links data: {user_links_array.shape}')
# pickle.dump(user_links_array, open(data_dir+'links_data_small.pkl','wb'))

user_links_array = pickle.load(open(data_dir+'links_data_small.pkl','rb'))
links_len = user_links_array.shape[1] - (user_links_array==-1).sum(axis=1)
logging.info(f"user links: max_links_len={links_len.max()}, avg links len = {links_len.mean()}, std links len={links_len.std()}")
user_links_array = user_links_array[:,:1800]
logging.info(user_links_array.shape)
pickle.dump(user_links_array, open(data_dir+'links_data_small_trunc.pkl','wb'))




# gt = pd.read_csv(data_dir+'hamas_reddit_gt_data.csv',lineterminator='\n')
# logging.info(f"gt shape {gt.shape}")
# idx_to_delete = gt.reset_index(drop=True)[(gt['Israel'].isnull()) | (gt['Palestine'].isnull())].index

# links = pickle.load(open(data_dir+'links_data.pkl','rb'))
# logging.info(f"links shape {links.shape}")
# links = np.delete(links,idx_to_delete,axis=0)
# logging.info(f"links shape {links.shape}")
# pickle.dump(links,open(data_dir+'links_data.pkl','wb'))

# gt = gt.drop(idx_to_delete)
# logging.info(f"gt shape {gt.shape}")
# pickle.dump(gt['Israel'].values,open(data_dir+'gt_data_Israel.pkl','wb'))
# pickle.dump(gt['Palestine'].values,open(data_dir+'gt_data_Palestine.pkl','wb'))