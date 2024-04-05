import os
from glob import glob
import pandas as pd
import numpy as np
from collections import Counter
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
from tweet_preprocessing import preprocess_tweet


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


user_tot_tweet_threshold = 20 # top 8% users

start_date = '2022-01-01'
end_date = '2023-01-01'
agg_time_period='3D'
entire_time_range = pd.date_range(start=start_date,end=end_date,freq=agg_time_period,tz='utc')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4096
MODEL = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

n_comp = 5

feat_cols = ['care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'anger', 'anticipation', 'disgust', 'fear', 'joy',
       'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# data_dir = '/nas/eclairnas01/users/siyiguo/rvw_data/rvw_us_en_data_mf_emot.pkl'
# df = pickle.load(open(data_dir,'rb'))
# logging.info(f"data loaded. size: {df.shape}, columns:\n{df.columns}")
# logging.info(f"min date:{df['created_at'].min()}, max date:{df['created_at'].max()}")
# # logging.info(f"languages:\n{df['tweet_language'].value_counts()}")

# df_scores = pickle.load(open('/nas/eclairnas01/users/siyiguo/rvw_data/ideology_scores.pkl','rb'))
# df = df.merge(df_scores,how='inner',left_on='author.username',right_on='screen_name')
# logging.info(f"after merging with ideology scores - shape:{df.shape},{len(pd.unique(df['author.username']))}")
# df.to_csv('/nas/eclairnas01/users/siyiguo/rvw_data/rvw_us_en_data_mf_emot_ideology.csv')

data_dir = '/nas/eclairnas01/users/siyiguo/rvw_data/rvw_us_en_data_mf_emot_ideology.csv'

df = pd.read_csv(data_dir,lineterminator='\n')

df['created_at'] = pd.to_datetime(df['created_at'],utc=True)
logging.info(f"min date:{df['created_at'].min()}, max date:{df['created_at'].max()}")

user_tweet_count = df.groupby('author.username')['id'].count()
logging.info(f"total num of users: {len(user_tweet_count)}")
logging.info(f"num tweets per user: mean={user_tweet_count.mean()}, std={user_tweet_count.std()}")
logging.info(f"0.5 quantile={user_tweet_count.quantile(q=0.5)}, 0.75 quantile={user_tweet_count.quantile(q=0.75)}, 0.8 quantile={user_tweet_count.quantile(q=0.8)}, 0.9 quantile={user_tweet_count.quantile(q=0.9)}")

active_users = user_tweet_count[user_tweet_count>=user_tot_tweet_threshold]
active_user_set = set(active_users.index)
df = df[df['author.username'].isin(active_user_set)]
df = df.reset_index(drop=True)
logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/len(user_tweet_count)}')
logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')

user_ts_count = df.groupby(['author.username',pd.Grouper(freq=agg_time_period,key='created_at')])['id'].count()
user_ts_count_ = user_ts_count.reindex(pd.MultiIndex.from_product([user_ts_count.index.levels[0],entire_time_range],names=['author.username','created_at']),fill_value=0)
logging.info(f"{agg_time_period} num tweets avg across days for each user. sum={user_ts_count_.sum()}. distr across users: mean={user_ts_count_.groupby(level=0).mean().mean()}, std={user_ts_count_.groupby(level=0).mean().std()}")
logging.info(f"0.5 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.5)}, 0.75 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.75)}, 0.8 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.8)}, 0.9 quantile={user_ts_count_.groupby(level=0).mean().quantile(q=0.9)}")


# ######################## BERT embedding features ########################
# # ts data with another set of features - bert embedding - umap reduced
# logging.info('start computing BERT embeddings')
# all_embeddings = np.empty((0,768))
# for i in tqdm(range(len(df)//batch_size+1)):
#     tmp = df[i*batch_size:(i+1)*batch_size]
#     tmp['text'] = tmp['text'].apply(preprocess_tweet)
#     encoded_input = tokenizer(tmp['text'].tolist(),max_length=50,truncation=True,padding=True,return_tensors='pt').to(device)
#     with torch.no_grad():
#         embeddings = model(**encoded_input).pooler_output
#     embeddings = embeddings.cpu().numpy()
#     all_embeddings = np.vstack((all_embeddings,embeddings))
#     if all_embeddings.shape[0] >= 5000000:
#         logging.info(f'saving intermediate BERT embeddings at batch={i}, embeddings shape: {all_embeddings.shape}')
#         pickle.dump(all_embeddings,open(f'/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_{i}.pkl','wb'))
#         all_embeddings = np.empty((0,768))

# logging.info(f'BERT embeddings finished, all_embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open(f'/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_{i}.pkl','wb'))
# logging.info('BERT embeddings saved.')

# all_embeddings = np.empty((0,768))
# emb_files = glob('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_*.pkl')
# for f in emb_files:
#     # logging.info(pickle.load(open(f,'rb')).shape)
#     all_embeddings = np.vstack((all_embeddings,pickle.load(open(f,'rb'))))
# logging.info(f'loaded saved bert embeddings, shape: {all_embeddings.shape}')

# # dim reduction - UMAP - OOM
# reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',verbose=True)
# all_embeddings = reducer.fit_transform(all_embeddings)
# logging.info(f'UMAP finshed, dimension reduced embeddings shape: {all_embeddings.shape}')

# dim reduction - pca
# logging.info('start PCA')
# # OOM - sample 50% data to train PCA and transform the rest
# idx = np.random.randint(len(all_embeddings), size=len(all_embeddings)//2)
# all_embeddings = all_embeddings[idx,:]
# std_scaler = StandardScaler()
# std_scaler = std_scaler.fit(all_embeddings)
# all_embeddings = std_scaler.fit_transform(all_embeddings)
# pickle.dump(std_scaler,open('/nas/eclairnas01/users/siyiguo/rvw_data/std_scaler_model.pkl','wb'))
# logging.info("standardized embeddings")

# reducer = PCA(n_components=n_comp)
# reducer = reducer.fit(all_embeddings)
# pickle.dump(reducer,open('/nas/eclairnas01/users/siyiguo/rvw_data/pca_model.pkl','wb'))
# logging.info('PCA model saved')

# std_scaler = pickle.load(open('/nas/eclairnas01/users/siyiguo/rvw_data/std_scaler_model.pkl','rb'))
# reducer = pickle.load(open('/nas/eclairnas01/users/siyiguo/rvw_data/pca_model.pkl','rb'))

# emb_files = glob('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_*.pkl')
# for f in emb_files:
#     # logging.info(pickle.load(open(f,'rb')).shape)
#     all_embeddings = pickle.load(open(f,'rb'))
#     all_embeddings = std_scaler.transform(all_embeddings)
#     all_embeddings = reducer.transform(all_embeddings)
#     pickle.dump(all_embeddings,open(f"{f.split('.')[0]}_pca.pkl",'wb'))
#     logging.info(f'saved bert embeddings pca, shape: {all_embeddings.shape}')

# all_embeddings = np.empty((0,5))
# pca_files =  glob('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_*_pca.pkl')
# for f in pca_files:
#     all_embeddings = np.vstack((all_embeddings,pickle.load(open(f,'rb'))))

# logging.info(f'PCA finshed, dimension reduced embeddings shape: {all_embeddings.shape}')
# pickle.dump(all_embeddings,open('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_pca.pkl','wb'))
# logging.info("PCA embeddings saved.")

all_embeddings = pickle.load(open('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_pca.pkl','rb'))
df[list(range(n_comp))] = all_embeddings

user_ts_data = df.groupby(['author.username',pd.Grouper(freq=agg_time_period,key='created_at')])[list(range(n_comp))+feat_cols].sum()
user_ts_data['tweet_count'] = df.groupby(['author.username',pd.Grouper(freq=agg_time_period,key='created_at')])['id'].count()
logging.info(f'raw user embedding ts data - shape: {user_ts_data.shape}')
# fill the time series with the entire time range
user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['author.username','created_at']),fill_value=0)
logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(pd.unique(user_ts_data.index.get_level_values(level=0)))}, len of entire time range: {len(pd.unique(user_ts_data.index.get_level_values(level=1)))}')

# transform into 3-d np array
ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
pickle.dump(ts_array[:,:,:-1], open('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_emotmf_ts_data_3D.pkl','wb'))
logging.info('finished saving BERT embeddings ts data')

# keep record to make sure users are indexed in the same order
ordered_user_index = user_ts_data.groupby(level=0)[0].first().index
logging.info(f"ordered_user_index:\n{list(ordered_user_index)[:10]}")

# select only a period of time
user_ts_data_ = user_ts_data[(user_ts_data.index.get_level_values(level=1)>=pd.Timestamp('2022-01-01',tz='utc'))&(user_ts_data.index.get_level_values(level=1)<pd.Timestamp('2022-05-02',tz='utc'))]
logging.info(f'user ts data in selected time range 2022-01-01 to 2022-05-02 - shape: {user_ts_data_.shape}; number of users: {len(pd.unique(user_ts_data_.index.get_level_values(level=0)))}, len of entire time range: {len(pd.unique(user_ts_data_.index.get_level_values(level=1)))}')
ts_array = np.array(user_ts_data_.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
pickle.dump(ts_array[:,:,:-1], open('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_emotmf_ts_data_0101_0502_3D.pkl','wb'))

user_ts_data_ = user_ts_data[(user_ts_data.index.get_level_values(level=1)>=pd.Timestamp('2022-06-24',tz='utc'))&(user_ts_data.index.get_level_values(level=1)<pd.Timestamp('2022-11-08',tz='utc'))]
logging.info(f'user ts data in selected time range 2022-06-24 to 2022-11-08 - shape: {user_ts_data_.shape}; number of users: {len(pd.unique(user_ts_data_.index.get_level_values(level=0)))}, len of entire time range: {len(pd.unique(user_ts_data_.index.get_level_values(level=1)))}')
ts_array = np.array(user_ts_data_.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
logging.info(f'shape of np array for the ts data: {ts_array.shape}, mean of embeddings: {np.mean(ts_array,axis=1)}')
pickle.dump(ts_array[:,:,:-1], open('/nas/eclairnas01/users/siyiguo/rvw_data/bert_embeddings_emotmf_ts_data_0624_1108_3D.pkl','wb'))

####################### demographic data ########################
# demo_cols = ['author.public_metrics.followers_count','author.public_metrics.following_count','author.public_metrics.tweet_count']

# missing_users = ['1776Jessi3CAgrl', '18MrGoat', '1ProudConsrv', '2legit2quit9526', '3141592653589R_', '45RapeKatieJohn', '4LiLs_Frankie', '4_FSake', '5thLegion_', '78Royaltree', '89Annli', '8kulou', 'AGFTRPOM', 'AJinMN', 'AMELIE_LUXURY', 'Ahsoka_ArtsTano', 'AmericanByGod', 'Amfirstnewswire', 'AndreaLWrites', 'AndrewJohnRosie', 'AngelicFusion', 'AnneofDerry', 'AnnisaDavenport', 'Anti45Potus2022', 'Apollonius_3', 'ArizFamHealth', 'Ashthebot_', 'BCUSA10', 'BambiNotGiving', 'BannedQuestion', 'Bartzma', 'Beautiful_Str34', 'Bella_2025z', 'Benen_M', 'BigChungoNS', 'BigNickR10', 'Billluna23', 'Biyatch777', 'BmcUcla', 'Bocinsky_Mary', 'BondGirlJinx', 'BooBooBeanx', 'BookSlutSasha', 'BowdenOnBass', 'Bradford1865', 'BrigidNyx', 'BrilLiScHiLL', 'BuckeyeTexan12', 'BusinessmanWiz', 'CPUSA_MI', 'Catfishgirl7373', 'Catholic_Scot', 'Cedrico_Suave', 'CeeMcGee63', 'ChelleStenn', 'ChrisCordani1', 'ChrisJo06211966', 'ChubbyHubby2U', 'CindyKNEW', 'CityPaperUSA', 'ClaraCatte', 'ColinStuckert', 'Commandowax', 'ConfusedbyLibs', 'CormorantSage', 'CreatureAuthor', 'Cuba_Sexualx', 'Cup_of_Jo__', 'Cynthia10171799', 'DEMSAREPATRIOTS', 'DJDreamware', 'DMRHerbs', 'DMRPoliticsCzar', 'DRWF7501', 'Danty99805058', 'DarkBlueKen', 'DaveMisura1', 'DavidStump77', 'DawnWalker1968', 'DemocracyVoterA', 'DesJones17', 'Dhavidey', 'DollarPamann201', 'DopeItsDomi_', 'DrazanParody', 'DriveByGeek', 'ERvettechmom', 'EarthFirst2022', 'EidolonOracle', 'Elvin_Unleashed', 'EpochTV', 'EricaOomed', 'ErinCSMH', 'EstablishCrat', 'ExplosiveSeal', 'F8thfulPolitics', 'F__L__Baker1997', 'FikaMaine', 'FinnTheGoodEgg', 'FluteTisa', 'Four50One', 'Fouraces007', 'FrancoMartinUSA', 'FreeSpeechTexan', 'FriendoftheCCP', 'FrontRowHope', 'FuefinaWG2058', 'GGCNJRun', 'GarethMcMullen', 'GaryPetersDem', 'GeminiUnlocked', 'GenXGemini', 'GhostAgent666', 'Gigi_4Trump', 'GingergirlGP', 'Gir_PupForm', 'GrammyVoteSmith', 'Grandma_Pixie', 'GreatLakePolitc', 'GuyYabera', 'HIBYETweet', 'HandyDandyDanD1', 'HavenDurant98', 'HeWhoReads62', 'HeartsOf_3', 'HeatHateMe', 'HenryLWalker', 'HokieMum', 'HonourableV2', 'HopeClinicWomen', 'INVResearchTeam', 'IamTheRebelLynn', 'ImAGhostToo', 'Independent74PA', 'Iron_Throne_', 'IronyisdeadLori', 'JJ_CMA_DCP_2022', 'JamesLe69338171', 'JanetScoopRedux', 'JayCantoni', 'JenieJes', 'JillFWood', 'JillyWitt', 'JimmyNeutron315', 'JoanneRNfromWI', 'JohnMC_MD', 'JohnS_Merc', 'JonathanCage10', 'JoshSessler', 'JstynSz', 'JustNanna04', 'JustVinnieD', 'KC_Democracy', 'KGBird32', 'KailinLembke', 'KansasGOPSlayer', 'KarlFetterwoman', 'KarlaforFlorida', 'KathyLDawn', 'KenLWaite', 'KevinCUniverse', 'Kevyna35', 'KommissarKrunch', 'Kristi_4_Toledo', 'KyleMasonMAGA', 'LSUVernal', 'LaJollaVB', 'LaOtraDarkOne', 'LarkinBain26', 'Latino921Lee', 'LaurenVegaNYC', 'LeeLandis72', 'LeftistsRDumdum', 'Legotrekker', 'Lenv777', 'LibertyGlens', 'LilDonnyB', 'LilMandee86', 'LitLoco4potus', 'LizzieIanthier', 'LkCumberlandKY2', 'Lollypops4us', 'Loren5thirteen', 'Lulitonet', 'MAGAretweets00', 'MDfromNY', 'MHMDmwalad1997_', 'MWebster__', 'MYOFBM', 'Mae_Ardsy', 'MattMojoMan', 'MattNyssen', 'MegaOtakuHS', 'MelodyM02997972', 'Michael67836612', 'MichelleTDTC', 'MikeVickNews', 'Mileycoddle', 'MitchytheKid66', 'Mooner14513', 'MrJPalm20', 'MrsErodvold1101', 'MyVotes4sale', 'NIPINTexas', 'NV_Editing', 'NYKrisL', 'NYSRepublican24', 'NanaShenanigans', 'NancyYoungblut', 'NatasjaYonce', 'NesEvan9', 'NettieBoivin3', 'NewYorkPopulist', 'Nick_nick5273', 'NimaSalis', 'NoBiasHeadlines', 'NoahRunsHisJaw', 'NotTommyThomas', 'NowHipHopNews_', 'OdotSchool', 'OhMyGumz', 'OnThisPain13', 'Onlymaybenate', 'OrneryCassandra', 'OutcastBrian', 'OvertIntent', 'OwnedByaLib', 'PACKERCENTURY', 'PHOENIX21PHOTON', 'PSLWitch', 'PalusRaluca', 'PatriarchParty', 'Patrice_Brost1', 'PatrickMusick', 'PatriotPstr55', 'Patti_M_Moss', 'PeachStatePulse', 'Peruvian_USA', 'Pharez85', 'Phil2Good2BTrue', 'Phil_Funnie', 'PoliticCulture', 'PonceyNAZ', 'PrinceSiix', 'PrinceofAmeri', 'Prj_Sentinel', 'QuecianaWalton_', 'RHYTHM313', 'RPFCstream', 'Rachel22Queen', 'RealCindyElla', 'RealJoeBonanno', 'RebeccaRSharp', 'Remy12231918', 'RepublicanCoryR', 'RevJasonStone', 'RichLiving1982', 'RisingUppercut8', 'Roberto02248702', 'RosebudAzn', 'Saba_6985', 'SadoLogan', 'SamsBadTweets', 'Sandywc2022', 'Santana28229861', 'Sarah_RN_WI', 'Satillaspring', 'ScottishKilt', 'SelfJim', 'SelfMadeOps', 'Seriously_Gr8t', 'SethoSteale', 'Shadow2b1', 'SharkGirl1973', 'ShellyTrueBlue', 'Shmoe45', 'ShotOut760', 'Sil_SEngle', 'SinisterLeftist', 'SisterDespair', 'SmedleyBUSMC', 'Smith0jade', 'SnowtailVeil', 'SoFloBullsFan', 'SocratesBigBird', 'SouFloCon', 'SoyruRyuko', 'SpiritualPagan', 'StaffordsElbow', 'StationOPJN', 'StepUpAmericans', 'StethoscopeOn', 'StopHangerBoys', 'SugarMoonDonuts', 'Sultan4Oz', 'SunshineRay59', 'SuperSw33tLu', 'Surly_Old_Man', 'SweetDanger69', 'TATPhotography', 'THEEangsiegel', 'TN_AndrewM', 'TS0828', 'TechradarJames', 'TheGlitterJojo', 'TheIrishDancer', 'TheJonSpikes', 'TheMatrixCBC', 'TheRewiredSoul', 'TheThundervamp9', 'TheWestGod', 'Thunderbuns54', 'Tim98915209', 'TimWSternberg', 'Tlkbouteverythg', 'ToddMxyzptlk', 'TormentedSocial', 'TraumazineDior', 'TruMomma3', 'TrumpBrothers', 'ULTRAredwave', 'UltraAntiMagat', 'UrsichJr', 'VeiledTigress', 'VetrepreneurOne', 'VoteTimWalzOut', 'VotingBlueIn22', 'WDOStairs', 'WesWordman', 'Wilfredmartin__', 'Will_MartinS22', 'Wolf88AM', 'XERALITA', 'Yongkinfan2021', 'ZonkerPA', '_24_mar_', '_ChristmasCat_', '_JoshDaugherty', '_Rile_E_Coyote', '_red_wave', '_shaaaun_', 'adore_leelee3', 'aimindmeld', 'alRiggsMusicOk', 'allieptweet', 'alwaysbmovin', 'andrewRoss_ny', 'annepowell1955', 'annoyedfword', 'artjackdreams', 'b_hors7STEM', 'babyvanorden', 'battlebornsfs', 'bbyxtaae', 'beansbeetsbikes', 'becMA_', 'blagogirl', 'bluefairyfly', 'bluenetwork12', 'blueslovr', 'bongsoverboys', 'breefatimuhhh', 'brett926171342', 'bubblerinny', 'bvsedkam', 'ca38745857', 'cambrierhodes', 'cammiebuckeye29', 'carlenebedient', 'carolinewe122', 'certifiedbbgal', 'chiamarc', 'chicalatina_', 'chrislejohnyc', 'chuck_vollmer', 'cjoblonskiwicz', 'classicplaygirl', 'clevebland', 'clptnfan', 'codyofthedead', 'conjornyn', 'constanceahath', 'cruzinbosco', 'cyrus333444', 'd_schweigert', 'daddy38456', 'daye_lovelyhori', 'democracywon22', 'diana31544504', 'dowellvalarie1', 'dredwards1028', 'drinkupb1tch', 'eIkgroves', 'edge_ofthe_west', 'ehelenes44', 'electron9_1', 'elliepheffernan', 'elroymush', 'enythe_green', 'ethanckelly', 'factfinds_24', 'fandomfanboi', 'fear4apricity', 'feministslikeus', 'fleet_detrik', 'fneudecker', 'freedom_1776_o7', 'fstarsjeweller', 'gabbyy21xo', 'gabmatic1990', 'globalvoicesltd', 'gobsmacked4real', 'goddesszer0', 'goldenrulepat', 'gracebutbetter', 'haospecial', 'harrahgirl', 'hasanpalacio', 'hoosier_victory', 'hope4change2022', 'horrorbiwriter', 'iChicosuave', 'iWealthConnect', 'igarglewithfire', 'igniteandlubs', 'immortalwarri0r', 'indexnforgetit', 'ismailkolya', 'itstresuree', 'jaclyninbklyn', 'jadasasuperstar', 'jadelesidi', 'jaxonxturner', 'jaypomYT', 'jeanniemarbuck', 'jeffery13131313', 'jenja38466728', 'jerrymaycares', 'jessica_anwyn', 'jillclauck', 'jjlopezlat', 'johnbuscharmi', 'johnchickenwin', 'jordan_NEhome', 'josh_g_28', 'joytmpsn', 'julieleach00011', 'jus_Heather', 'k_TAP_g', 'kaleymaemae', 'karissaxrose', 'katearthsis', 'keepinupwij_', 'kevinkunze__', 'kluehlj', 'knitssocks', 'krixter59', 'kyicon_', 'lataet', 'lawbrarianstorm', 'lctuckerHLSTN', 'leftistlangcat', 'leftwingecho', 'lezxalejandraaa', 'librasgroooove', 'lickmycoconut', 'lmmswg', 'louanna850', 'maeismighty', 'magastorm24', 'manicclementine', 'mariah258307062', 'markets_mt', 'mauri_vista', 'mbodolay99', 'mcwexler_jane', 'mdeluca007', 'meTracyEastwood', 'melisathepisces', 'mfcmnapl', 'michell41527274', 'midthirty3', 'mindfulmimosa', 'miraisandrist', 'mms5048', 'moonstoneaura13', 'morebluevotes', 'mp_hicks', 'mrsastrologer', 'natss_day', 'nextlevelbi', 'nic_pul', 'nlg444', 'nofrackingcalco', 'nomor3pai', 'notpincheblanco', 'novembergoth5', 'nradd2020', 'obaroag2', 'ogjp1975', 'onebraverifle2', 'onebuddhistpunk', 'orangegod2017', 'otaimango', 'outlawJoZ13', 'patriot1776va', 'peterunion75', 'pfaff4congress', 'pickyoursetlist', 'pizzashoem1', 'pjitsjustpj', 'pollypockit', 'pslmpls', 'puh_nayproud', 'rahbeyondman', 'rayarighteous', 'realNUCLEARMAGA', 'redNYer', 'roanneaz63', 'rosiebackroads', 'roxinbound', 'ryand4ry', 'sabrozowski22', 'saltwater_sand', 'saveourdemoc72', 'scottlawrencenh', 'sebaaide', 'seemseem222', 'sgarrettpate', 'sharoneducator', 'shd_val_atheist', 'shreksthot', 'sigarettemom', 'simone_taline', 'soundnessmind', 'stephc1024', 'stridentTH', 'sunshinektx67', 'suzyjones555', 'svenix1yngle', 'swannsong82', 'symppho_', 'tabascopowered', 'taguedawg', 'tapestrymedia_', 'tehrooniii', 'tg_ghoul_lady', 'thatdalugowski', 'thebluestarz', 'thecherrymoon__', 'thecheyvyshnya', 'theoScotK', 'tjy_mich', 'tonganbassbruja', 'toxicth0mas', 'transHypatia', 'witchmazikeen', 'wokewokitywoke', 'xltiktokcia', 'xojadeeey', 'xorochelle23', 'xtinascialoia_', 'yoyoyupyupmutha']

# demo = pd.read_csv('/nas/eclairnas01/users/siyiguo/rvw_data/rvw_user_attributes.csv',lineterminator='\n')
# demo = demo.set_index('author.username')
# for u in missing_users:
#     demo.loc[u,demo_cols] = 100

# demo = demo.loc[ordered_user_index,demo_cols]
# logging.info(f"demo data shape={demo.shape}")
# pickle.dump(demo.values,open('/nas/eclairnas01/users/siyiguo/rvw_data/demo_data.pkl','wb'))

# ####################### ground truth data - political ideology ########################
# # get ground truth data
# gt = df.groupby('author.username')['political_gen'].first().loc[ordered_user_index,] # 0 = left, 1 = right
# logging.info(f"gt shape={gt.shape}, num left users={len(gt[gt==0])}, num right users={len(gt[gt==1])}")
# logging.info(f"gt author order:\n{list(gt.index)[:10]}")
# # pickle.dump(gt.values,open('/nas/eclairnas01/users/siyiguo/rvw_data/gt_data.pkl','wb'))
# # logging.info('finished saving ground truth data')

# # elites data - for training *** <0 is lib >0 is con ***
# df_elites = pd.read_csv('/nas/eclairnas01/users/siyiguo/ts_clustering/data/elites_twitter_political.tsv',delimiter='\t')
# df_elites = df_elites[~df_elites['phi'].isnull()]
# logging.info(f"elites df shape={df_elites.shape}")
# gt = gt.reset_index().reset_index()
# df_elites = df_elites.merge(gt,how='inner',left_on='screen_name',right_on='author.username')
# logging.info(f"merge gt and df_elites, shape={df_elites.shape}\n{df_elites.columns}")
# df_elites['elite_label'] = 0
# df_elites.loc[df_elites['phi']>0,'elite_label'] = 1
# logging.info(f"orig phi: #lib={len(df_elites[df_elites['phi']<0])}, elite_label: #lib={len(df_elites[df_elites['elite_label']==0])}")
# logging.info(f"orig phi: #con={len(df_elites[df_elites['phi']>0])}, elite_label: #con={len(df_elites[df_elites['elite_label']==1])}")
# logging.info(f"label & elite_label discrepancy: {(df_elites['elite_label']!=df_elites['political_gen']).sum()}")
# df_elites.to_csv('/nas/eclairnas01/users/siyiguo/ts_clustering/data/rvw_elites_twitter_political.tsv')
# logging.info(df_elites['index'].values)

# # there are discrepancies between ashwin's prediction and elite's political ideology. change these in the gt data.
# gt_ = gt.merge(df_elites[['screen_name','elite_label']],how='left',left_on='author.username',right_on='screen_name')
# gt_ = gt_.set_index('author.username')
# gt_ = gt_.loc[ordered_user_index,]
# gt_.loc[~gt_['elite_label'].isnull(),'political_gen'] = gt_.loc[~gt_['elite_label'].isnull(),'elite_label']
# logging.info(f"gt before saving - gt shape={gt_.shape}, num left users={len(gt_[gt_['political_gen']==0])}, num right users={len(gt_[gt_['political_gen']==1])}")
# logging.info(f"gt author order:\n{list(gt_.index)[:10]}")
# pickle.dump(gt_.political_gen.values,open('/nas/eclairnas01/users/siyiguo/rvw_data/gt_data.pkl','wb'))
# gt_.to_csv('/nas/eclairnas01/users/siyiguo/rvw_data/gt_data.csv')
# logging.info('finished saving gt data')




"""
Index(['id', 'author.username', 'created_at', 'text', 'entities.urls',
       'entities.hashtags', 'lang', 'source', 'author.id', 'author.created_at',
       'author.username', 'author.name', 'author.verified', 'stance', 'date',
       'str_date', 'loc', 'us_flag', 'state', 'care', 'harm', 'fairness',
       'cheating', 'loyalty', 'betrayal', 'authority', 'subversion', 'purity',
       'degradation', 'anger', 'anticipation', 'disgust', 'fear', 'joy',
       'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 
       'screen_name', 'pol_proba', 'political_gen'],
      dtype='object')
"""

"""
elites indices in gt data:
[ 55026,  32302, 116711,  55009, 115248,  82134,  54991,  12019,
        41424,  55021,  39729,  54999,  55033,  55011,  30263,  64876,
        55020,  88313,  51210,  51238,  51181,   7269,  55007,  51193,
        51217,  36660,  51220,  10790,  51206,  55030,  51228,  51234,
        54203,  51219,  13411,  51201,  51188,  51229,   5332,  26048,
        39047,  51180,  51235,  51237,  51189,  51197,  51195, 118737,
        51192,  54825,  51190,  51208,  51207,  54998,  52439,  13210,
        51231,  51973,  51225,  51203,  48845,  51240,  51179,  51194,
        51178,  51185,  51216, 111133, 115263,  52853,  29920,  51212,
        51187,  51236,  81796,  51182,  51204,  69912,  51202,  51199,
        51213, 108006,  79251,  51215,  51183,  84107,  54997,  51243,
        98906,  51200,  61331,  12132,   6934,  21464,  56890,  55022,
        35717,  55006,  18859,  55002,  22376,  26817,  23349,  84918,
        61347, 114975,  23347,  57591,  23344,  23354,  23339,  10490,
        55035, 103708, 119338,  64316,  43773,   9587, 106035, 115991,
        94969,  62829,   9878,   9871,  21255,  44818, 103701,  44897,
       103514,   1427, 106431,  45727,  14796, 115879,   9371, 115815,
       102786, 116008,  44105,  44860, 116139,   1428,  72893, 109644,
        93107, 109653, 103049,  78040,  94410,  75348,  74636,  88691,
        95719,  99363, 104109,  76906,  98773,  90845,  98719,  90638,
        90810,  78272,  72814,  68723, 110051,  72861, 103139, 105059,
        68816,  77918, 101801,  91019, 111143, 109899,  67113, 114134,
       110303,  69168, 102629, 105076,  91576, 112741,  71599,  83828,
        94434, 119991,  71870,  80926,  69820,  83367,  72826, 100824,
        69319, 108380,  70648,  90803,  67504,  97584,  98311, 108376,
        68707, 115960,  75327,  89832,  85171,  92417,  98081, 106033,
       110916,  90330, 100692,  86526,  87129,  92924, 105021, 113635,
        74851,  99872,  90464,  99560, 101634, 113459,  90596, 120530,
       119919, 108747, 117478,  14674,  43682,  48669,  58543,  30299,
       103414,  60631,  57552,   6885,  61330,   9726,  30292,   9552,
        32300,   6564,  12357,  74125, 106045,   9060,  23342,  82984,
        36406,  60166,  85518,   3492,   5351,  45818, 101214,  47306,
        71161,  49341,  44592,  98871,  34819,  44816,  25065, 111216,
        84479,  45368, 100810,   2261,  48667,  31187, 105963,   2315,
         9269, 113314, 109619,  59426,  59448,  89732,    947,  74815,
         3264,  67905,  19752,  90829,  15830,  98305,  23623,  97119,
        77524,  47045,   9163,  97025,  99375,  96038,  99444,  47320,
        81162,  91770,  83135,  90922,  98689,  14846,  20419, 111054,
        67365,  14536,  93199,  75190,  45121, 112762,  55210,   2614,
         2130,  91767,  87081,  38577,  45508,  26127,   7329,  44483,
       109595, 102609,  23333,   5552,  58561,  81574, 100094,  49867,
       116114,  68601,  29035, 101995,   5876,  94462, 112684,  41888,
        15728,   8179,  42253,   8798,  55481,  19213,  23410,  50556,
        87418,  60169, 101519,  71337,   3881,  69417,  30572,  12695,
        66166,  16732,  84761,  68031,  64506, 109039]
"""