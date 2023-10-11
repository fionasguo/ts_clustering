from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 5
CROP_TO = 32
SEED = 26

PROJECT_DIM = 2048
LATENT_DIM = 512
WEIGHT_DECAY = 0.0005

# Ceate a cosine decay learning scheduler.
num_training_samples = len(x_train)
steps = EPOCHS * (num_training_samples // BATCH_SIZE)
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.03, decay_steps=steps
)

# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)

# Compile model and start training.
simsiam = SimSiam(get_encoder(), get_predictor())
simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
history = simsiam.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])

# Visualize the training progress of the model.
plt.plot(history.history["loss"])
plt.grid()
plt.title("Negative Cosine Similairty")
plt.show()

##############################################################

repeats = {k:10 for k in [10,20,30,40,50,60]}
lds = [10,20,30,40,50]
batch_size, lr, patience = 32, 0.0005, 10
d, N, he, dropout = 50,2,4,0.2
# fore_savepath = 'mimic_iii_24h_strats_no_interp_with_ss_fore.h5'
f = open('log.csv', 'a+')
f.write('\nTraining on different % of labeled data\n')

train_inds = np.arange(len(train_op))
valid_inds = np.arange(len(valid_op))
gen_res = {}

np.random.seed(2021)
for ld in lds:
    np.random.shuffle(train_inds)
    np.random.shuffle(valid_inds)
    train_starts = [int(i) for i in np.linspace(0, len(train_inds)-int(ld*len(train_inds)/100), repeats[ld])]
    valid_starts = [int(i) for i in np.linspace(0, len(valid_inds)-int(ld*len(valid_inds)/100), repeats[ld])]
    f.write('Training on '+str(ld)+' % of labaled data+\n'+'val_metric,roc_auc,pr_auc,min_rp,savepath\n')
    all_test_res = []
    for i in range(repeats[ld]):
        print ('Repeat', i, 'ld', ld)
        # Get train and validation data.
        curr_train_ind = train_inds[np.arange(train_starts[i], train_starts[i]+int(ld*len(train_inds)/100))]
        curr_valid_ind = valid_inds[np.arange(valid_starts[i], valid_starts[i]+int(ld*len(valid_inds)/100))]
        curr_train_ip = [ip[curr_train_ind] for ip in train_ip]
        curr_valid_ip = [ip[curr_valid_ind] for ip in valid_ip]
        curr_train_op = train_op[curr_train_ind]
        curr_valid_op = valid_op[curr_valid_ind]
        print ('Num train:',len(curr_train_op),'Num valid:',len(curr_valid_op))
        # Construct save_path.
        savepath = 'new_mimic_iii_24hm_strats_no_interp_with_ss_repeat'+str(i)+'_'+str(ld)+'ld'+'.h5'
        print (savepath)
        # Build and compile model.
        model, fore_model =  build_strats(D, max_len, V, d, N, he, dropout, forecast=True)
        model.compile(loss=mortality_loss, optimizer=Adam(lr))
        fore_model.compile(loss=forecast_loss, optimizer=Adam(lr))
        # Load pretrained weights here.
        fore_model.load_weights(fore_savepath)
        # Train model.
        es = EarlyStopping(monitor='custom_metric', patience=patience, mode='max', 
                           restore_best_weights=True)
        cus = CustomCallback(validation_data=(curr_valid_ip, curr_valid_op), batch_size=batch_size)
        his = model.fit(curr_train_ip, curr_train_op, batch_size=batch_size, epochs=1000,
                        verbose=1, callbacks=[cus, es]).history
        model.save_weights(savepath)
        # Test and write to log.
        rocauc, prauc, minrp = get_res(test_op, model.predict(test_ip, verbose=0, batch_size=batch_size))
        f.write(str(np.min(his['custom_metric']))+str(rocauc)+str(prauc)+str(minrp)+savepath+'\n')
        print ('Test res', rocauc, prauc, minrp)
        all_test_res.append([rocauc, prauc, minrp])
        
    gen_res[ld] = []
    for i in range(len(all_test_res[0])):
        nums = [test_res[i] for test_res in all_test_res]
        gen_res[ld].append((np.mean(nums), np.std(nums)))
    print ('gen_res', gen_res)
f.close()