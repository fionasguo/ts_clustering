import sys
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from preprocessing import preprocess_tweet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2048
MODEL = 'sentence-transformers/stsb-xlm-r-multilingual' # 'cardiffnlp/twitter-xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)


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


if __name__=='__main__':
    data_dir = sys.argv[1] # country dir

    df = pd.read_csv(data_dir+'/all_controls_drivers.csv',lineterminator='\n')

    logging.info('start computing XLMT embeddings')
    
    all_embeddings = np.empty((0,768))
    for i in tqdm(range(len(df)//batch_size+1)):
        tmp = df[i*batch_size:(i+1)*batch_size]
        tmp['tweet_text'] = tmp['tweet_text'].apply(preprocess_tweet)
        encoded_input = tokenizer(tmp['tweet_text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
        with torch.no_grad():
            embeddings = model(**encoded_input).pooler_output
        embeddings = embeddings.cpu().numpy()
        all_embeddings = np.vstack((all_embeddings,embeddings))

        if all_embeddings.shape[0] >= 5000000:
            logging.info(f'saving intermediate BERT embeddings at batch={i}, embeddings shape: {all_embeddings.shape}')
            pickle.dump(all_embeddings,open(f"{data_dir}/xlmt_embeddings_{i}.pkl",'wb'))
            all_embeddings = np.empty((0,768))
    
    logging.info(f'XLMT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
    pickle.dump(all_embeddings,open(f"{data_dir}/xlmt_embeddings_{i}.pkl",'wb'))
    logging.info('XLMT embeddings saved.')
    