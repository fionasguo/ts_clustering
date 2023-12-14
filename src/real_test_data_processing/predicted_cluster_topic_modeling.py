import os
from datetime import datetime
import pickle
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from preprocessing import preprocess_tweet
import logging
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('french'))


tm_save_dir = '/nas/home/siyiguo/ts_clustering/data/tmp_topic_models/'


def create_logger():
    try:
        logfilename = os.environ["SLURM_JOB_ID"]
    except:
        logfilename = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=logfilename + '.log',
                        format="%(message)s",
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True


def perform_bertopic(tweets, timestamps, fname):
    # get global topic model
    print('start topic modeling')
    try:
        sentence_model = SentenceTransformer('dangvantuan/sentence-camembert-large')
        dim_model = PCA(n_components=5)
        cluster_model = KMeans(n_clusters=50)
        topic_model = BERTopic(embedding_model=sentence_model, umap_model=dim_model, hdbscan_model=cluster_model,verbose=False)
        _,_ = topic_model.fit_transform(tweets)

        topic_model.save(tm_save_dir+'tm_'+fname, save_embedding_model=False)
        print('global topic model saved')

        # dynamic topic modeling
        topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)
        with open(tm_save_dir+'dtm_'+fname+'.pkl','wb+') as f:
            pickle.dump(topics_over_time,f)
        print('dynamic topic modeling done')

        logging.info(f'len of data: {str(len(topic_model.probabilities_))}')
        logging.info(f'number of topics: {str(len(topic_model.topic_labels_))}')
        # ftopics.write(topic_model.get_topics())
        for key, value in topic_model.get_topics().items(): 
            if key >=0 and key <= 20:
                logging.info(f'{key}:{[i[0] for i in value if i[0] not in stop_words]}\n')
        logging.info('\n')
    except Exception as e:
        print(fname)
        print(e)
        print('topic modeling failed!')
        return None

create_logger()

user_tot_tweet_threshold = 10 # only take users who has more than 10 tweets in about 90 days
start_date = '2017-03-01'
end_date = '2017-06-01'
agg_time_period='12H'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size = 2048
# MODEL = 'Yanzhu/bertweetfr-base'
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModel.from_pretrained(MODEL).to(device)

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

# df with gt and predicted clusters for each user
df_preds = pd.read_csv('/nas/home/siyiguo/ts_clustering/test_phase1a_bert_pca_midnoise/user_gt_preds.csv')
df_preds = df_preds[df_preds['label']==0] # only look at the biggest connected components in ground truth
logging.info(f'total number of users: {len(df_preds)}')

df = df.merge(df_preds,how='right',on='twitterAuthorScreenname')
logging.info(f'total number of tweets: {df.shape}')

logging.info('preprocessing texts')
tqdm.pandas()
df['contentText'] = df['contentText'].apply(preprocess_tweet)
logging.info('finished preprocessing texts')

pred_cluster_lst = list(pd.unique(df['preds']))
for pred_c in pred_cluster_lst:
    logging.info(f"predicted cluster {pred_c}")

    tmp = df[df['preds']==pred_c]
    logging.info(f"len of data: {tmp.shape}")

    perform_bertopic(tmp['contentText'].to_list(),tmp['timePublished'].to_list(),f"pred_cluster_{pred_c}")
