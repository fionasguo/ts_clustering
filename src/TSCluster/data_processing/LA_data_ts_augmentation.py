
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


all_cols = ['time', 'user_screen_name', 'user_id', 'user_verified',
       'user_followers_count', 'user_friends_count', 'user_favourites_count',
       'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
       'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'count']
user_attr_cols = ['user_screen_name', 'user_id', 'user_verified',
       'user_followers_count', 'user_friends_count', 'user_favourites_count']
emot_mf_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
       'pessimism', 'sadness', 'surprise', 'trust', 'care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'count']

full_daterange = pd.date_range(start='2020-01-01',end='2021-01-01',freq='H',tz='US/Pacific')


def transform_to_sparse_data_format(ts, cols):
    """
    transform data from pandas dataframe to list of tuples (time, feature name, value)

    ts: pd.DataFrame, index is time, column is user attributes + number of tweets with emot/mf per hour
    cols: the column names needs to be augmented
    return - result: list of tuples (time, variable, value)
    """
    result = []
    for c in cols:
        tmp = ts[c]
        tmp = tmp[tmp != 0]
        result.extend([(i[0].strftime('%Y-%m-%d:%H:%M:%S'),c,i[1]) for i in list(tmp.items())])
    return result


## generate positive examples by bootstrapping
def generate_positive_example(ts, cols, seed=3, debug=False):
    """
    generate one positive example using bootstrapping

    ts: pd.DataFrame, index is time, column is user attributes + number of tweets with emot/mf per hour
    cols: the column names needs to be augmented
    seed: random seed
    debug: bool, whether to print and plot
    return - results: list of tuples (time, variable, value)
    """
    result = []

    for c in cols:
        tmp = ts[c]
        tmp = tmp[tmp != 0]

        # what the original looks like
        if debug:
            print(f'{ts.user_screen_name.iloc[0]} - {c}')
            print(f'original data: len of data={len(tmp)}, sum={tmp.sum()}')
            # plot
            tmp.reindex(full_daterange).fillna(0).plot(figsize=[6,3])
            plt.legend()
            plt.title(f'original data - {ts.user_screen_name.iloc[0]} - {c}')
            plt.show()

        if len(tmp) <= 5:
            random.seed(random.choice(list(range(100000))))
            # randomly add 1 to 5 signals
            random_time_points = random.choices(list(full_daterange),k=random.choice(list(range(0,6,1))))
            if debug:
                print(f'create positive sample with {len(random_time_points)} points')
            result.extend([(i.strftime('%Y-%m-%d:%H:%M:%S'),c,1) for i in random_time_points])
        else:
            # bootstrapping
            random.seed(seed)
            btstrp_range = list(range(int(len(tmp)*0.7), int(len(tmp)*1.3)+2, 1)) # a range of the number of data points as the output of bootstrapping, e.g. original data have 50 data points, we can bootstrap 45 data points, or 55.
            btstrpd = random.choices(list(tmp.items()),k=random.choice(btstrp_range)) 
            btstrpd = pd.Series([i[1] for i in btstrpd],index=[i[0] for i in btstrpd])
            btstrpd = btstrpd.groupby(level=0).sum()
            if debug:
                print(f'bootstrapped data: len of data={len(btstrpd)}, sum={btstrpd.sum()}')
                # plot
                btstrpd.reindex(full_daterange).fillna(0).plot(figsize=[6,3])
                plt.legend()
                plt.title(f'bootstrapped data - {ts.user_screen_name.iloc[0]} - {c}')
                plt.show()
            
            result.extend([(i[0].strftime('%Y-%m-%d:%H:%M:%S'),c,i[1]) for i in list(btstrpd.items())])
    
    return result


## generate negative example by randomly sample another user's ts
def generate_negative_example(df, cols, user_name, seed=3):
    """
    generate one negative example by randomly selecting other user's ts from the overall pool

    df: pd.Dataframe, include per user per hour aggregation of emot and mf for all users
    cols: the column names needs to be augmented
    user_name: target user
    seed: random seed
    return - results: list of tuples (time, variable, value)
    """
    random.seed(seed)
    random_user_name = random.choice([i for i in list(pd.unique(df.user_screen_name)) if i!=user_name])
    tmp = df[df.user_screen_name==random_user_name].set_index('time')
    return transform_to_sparse_data_format(tmp, cols)

        

if __name__ == "__main__":
    partition = int(sys.argv[1])
    n_positive = 5 # number of positive examples to generate for each user
    n_negative = 5 # number of negative examples to generate for each user

    original_data = [] # key - user name, value: list of tuples (time, feature name, value)
    positive_examples = [] # key - user name, value: list of list of tuples (multiple positive examples)
    negative_examples = []

    ## load aggregated data (emotion/mfs per hour per user) for active users (20 or more tweets in 2020)
    active_user_df = pd.read_csv('/nas/home/siyiguo/user_similarity/data/LA_tweets_emot_mf_active_user_hourly_ts.csv')
    active_user_df['time'] = pd.to_datetime(active_user_df['time'])
    print(f'len of active_user_df = {len(active_user_df)}')

    # user list
    with open('/nas/home/siyiguo/user_similarity/data/active_user_list.txt','r') as f:
        user_list = f.read().splitlines() # list(pd.unique(active_user_df.user_screen_name))
    print(f'num of users={len(user_list)}') # should be 96787
    user_list_partition = user_list[partition*10000:(partition+1)*10000]

    cnt = 0
    for user in tqdm(user_list_partition): # for each user
        # transform original data to sparse data format
        tmp = active_user_df[active_user_df.user_screen_name==user].set_index('time')
        original_data.append({'user_screen_name':user, 'original':transform_to_sparse_data_format(tmp,cols=emot_mf_cols)})
        # positive examples
        positive_example = {'user_screen_name':user, 'positive':[]}
        for i in range(n_positive):
            positive_example['positive'].append(generate_positive_example(tmp, emot_mf_cols, seed=i, debug=False))
        positive_examples.append(positive_example)
        # negative examples
        negative_example = {'user_screen_name':user, 'negative':[]}
        for j in range(n_negative):
            negative_example['negative'].append(generate_negative_example(active_user_df, emot_mf_cols, user, seed=j))
        negative_examples.append(negative_example)

        cnt += 1
        if cnt % 50 == 0:
            with open('/nas/home/siyiguo/user_similarity/data/original_data_active_users_sparse_'+str(partition)+'.json','a+') as f:
                f.write(json.dumps(original_data))
            with open('/nas/home/siyiguo/user_similarity/data/positive_examples_active_users_sparse_'+str(partition)+'.json','a+') as f:
                f.write(json.dumps(positive_examples))
            with open('/nas/home/siyiguo/user_similarity/data/negative_examples_active_users_sparse_'+str(partition)+'.json','a+') as f:
                f.write(json.dumps(negative_examples))
            original_data = []
            positive_examples = []
            negative_examples = []
            

    with open('/nas/home/siyiguo/user_similarity/data/original_data_active_users_sparse_'+str(partition)+'.json','a+') as f:
        f.write(json.dumps(original_data))
    with open('/nas/home/siyiguo/user_similarity/data/positive_examples_active_users_sparse_'+str(partition)+'.json','a+') as f:
        f.write(json.dumps(positive_examples))
    with open('/nas/home/siyiguo/user_similarity/data/negative_examples_active_users_sparse_'+str(partition)+'.json','a+') as f:
        f.write(json.dumps(negative_examples))