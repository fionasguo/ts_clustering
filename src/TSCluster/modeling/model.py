import logging
import numpy as np
from itertools import combinations

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda, Add, Concatenate, ReLU, BatchNormalization
from tensorflow.keras import regularizers, Model

from .layers import CVE, Attention, Transformer


def get_encoder(
        n_feat: int,
        emb_dim: int = 50, 
        max_triplet_len: int = 200, 
        n_transformer_layer: int = 2, 
        n_attn_head: int = 4, 
        dropout: float = 0.2, 
        demo_dim: int = 0,
        tau: float = 1.0,
        lam: float = 1.0
    ):
    """
    The encoder that embed time series triplets into a vector.

    Params:
        n_feat: number of classes of feature dummies
        emb_dim: dimension for embedding vectors
        max_triplet_len: max length of triplets for each user
        n_transformer_layer: number of transformer layers
        n_attn_head: number of transformer attention heads
        dropout: dropout rate
        demo_dim: dim of demographic vector if zero means we don't have it
        tau: scalar that the timestamp embedding is multiplied with, controls how much timestamp contributes to the final representation
        lam: scalar that the feat and value embedding is multiplied with, controls how much features and values contributes to the final representation

    Return:
        embedding vectors
    """
    # _emb_dim = emb_dim * 2

    # embed feature dummy, or variable name
    varis = Input(shape=(max_triplet_len,))
    varis_emb = Embedding(n_feat, emb_dim)(varis) # output shape (batch size, max_triplet_len_triplets, d) # n_feat + 1 for unknown - not needed

    # embed timestamps and values
    values = Input(shape=(max_triplet_len,))
    times = Input(shape=(max_triplet_len,))
    cve_units = int(np.sqrt(emb_dim))
    values_emb = CVE(cve_units, emb_dim)(values) # output shape (batch size, max_triplet_len_triplets, d)
    times_emb = CVE(cve_units, emb_dim)(times)

    # multiply times embedding by a scalar before adding -> control how much of temporal info we want to learn
    times_emb = Lambda(lambda x: x * tau)(times_emb)
    varis_emb = Lambda(lambda x: x * lam)(varis_emb)
    values_emb = Lambda(lambda x: x * lam)(values_emb)

    # combine all embedded vectors
    emb = Add()([varis_emb, values_emb, times_emb]) # (batch size, max_triplet_len_triplets, d)
#     demo_enc = Lambda(lambda x:K.expand_dims(x, axis=-2))(demo_enc) # b, 1, d
#     comb_emb = Concatenate(axis=-2)([demo_enc, comb_emb]) # b, L+1, d
    
    # input vectors are padded with zero to max_triplet_len_triplets, build a mask to tell which ones are not observed
    # bound feat dummies between 0 and 1,feat dummy start from 1, so 0 means unobserved.
    mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L  
#     mask = Lambda(lambda x:K.concatenate((K.ones_like(x)[:,0:1], x), axis=-1))(mask) # b, L+1
    
    # go through transformer
    emb = Transformer(n_transformer_layer, n_attn_head, dk=None, dv=None, dff=None, dropout=dropout)(emb, mask=mask)
    # compute attention weights
    attn_weights = Attention(2*emb_dim)(emb, mask=mask)
    # fusion
    emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([emb, attn_weights])

    demo = Input(shape=(demo_dim,))
    if demo_dim > 0:
        # embed demographic vector - 2 layers of FFN
        
        demo_enc = Dense(2*emb_dim, activation='tanh')(demo)
        demo_enc = Dense(emb_dim, activation='tanh')(demo_enc)
        # concat to demographic embedding
        emb = Concatenate(axis=-1)([emb, demo_enc])

    emb = Dense(emb_dim,name='embed_output')(emb) # for forcasting
    # op = Dense(1, activation='sigmoid')(logit_op)

    model = Model({'demo':demo, 'timestamps':times, 'values':values, 'feat':varis}, emb, name = 'encoder')
    
    # if forecast:
    #     fore_model = Model([demo, times, values, varis], logit_op)
    #     return [model, fore_model]
    
    return model


def get_predictor(input_dim, hid_dim, weight_decay):
    """
    Predictor is an AutoEncoder type of structure, 
    input the embedding from encoder, and output to embeddings with same input_dim
    """
    # model = tf.keras.Sequential(
    #     [
    #         # Note the AutoEncoder-like structure.
    #         Input((input_dim,)),
    #         Dense(
    #             hid_dim,
    #             use_bias=False,
    #             kernel_regularizer=regularizers.l2(weight_decay),
    #         ),
    #         ReLU(),
    #         BatchNormalization(),
    #         Dense(input_dim),
    #     ],
    #     name="predictor",
    # )

    input_vec = Input((input_dim,))
    x = Dense(
                hid_dim,
                use_bias=False,
                kernel_regularizer=regularizers.l2(weight_decay),
            )(input_vec)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dense(input_dim)(x)

    model = Model(input_vec, x, name='predictor')

    return model

def get_link_predictor(input_dim, hid_dim, weight_decay):
    emb1 = Input((input_dim,))
    emb2 = Input((input_dim,))
    x = Concatenate(axis=-1)([emb1, emb2])

    # x = Dense(
    #             hid_dim*2,
    #             use_bias=False,
    #             kernel_regularizer=regularizers.l2(weight_decay),
    #         )(x)
    # x = ReLU()(x)
    # x = BatchNormalization()(x)
    x = Dense(
                hid_dim,
                use_bias=False,
                kernel_regularizer=regularizers.l2(weight_decay),
            )(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dense(1,activation='sigmoid')(x)

    model = Model((emb1,emb2), x, name='link_predictor')
    return model

def make_combinations(array,batch_size):
    size = tf.shape(array)[0]
    # Make 2D grid of indices
    r = tf.range(size)
    ii, jj = tf.meshgrid(r, r, indexing='ij')
    # Take pairs of indices where the first is less or equal than the second
    m = ii < jj
    m.set_shape([batch_size,batch_size])
    output_size = int((batch_size-1)/2*batch_size)
    return tf.gather(array, tf.reshape(tf.boolean_mask(ii, m),[output_size])),tf.gather(array,tf.reshape(tf.boolean_mask(jj, m),[output_size]))

def prepare_link_data(emb,indices,links,batch_size):
    """
    make pair combinations of all rows in emb, and label them with 1 if there's a link exists between them

    emb: tensor batch_size * hidden_dim
    indices: the index of each emb, (batch_size,)
    links: for each emb, the indicies of other embs that have a link with, batch_sizw * max # links (padded with -1)
    """
    emb1, emb2 = make_combinations(emb,batch_size)
    idx, tgt_idx = make_combinations(indices,batch_size)
    idx_to_links = tf.where(tf.equal(indices,idx[...,None]))[:,-1]
    expanded_links = tf.gather(links,indices=idx_to_links)
    y_link = tf.cast(tf.reduce_any(tf.cast(tf.equal(tgt_idx[...,None],expanded_links),tf.bool),axis=1),dtype=tf.float32)

    return emb1,emb2,y_link
    


class SimSiam(Model):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = get_encoder(
            n_feat=self.args['n_feat'],
            emb_dim=self.args['embed_dim'],
            max_triplet_len=self.args['max_triplet_len'], 
            n_transformer_layer=self.args['n_transformer_layer'], 
            n_attn_head=self.args['n_attn_head'], 
            dropout=self.args['dropout'], 
            demo_dim=self.args['demo_dim'],
            tau=self.args['tau'],
            lam=self.args['lam']
        )
        self.predictor = get_predictor(
            input_dim=self.args['embed_dim'],
            hid_dim=self.args['embed_dim'], 
            weight_decay=self.args['weight_decay']
        )
        self.link_predictor = get_link_predictor(
            input_dim=self.args['embed_dim'],
            hid_dim=self.args['embed_dim'], 
            weight_decay=self.args['weight_decay']
        )

        self.loss_fn = self.args['loss_fn']

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_SimSiam_loss(self, p, z):
        # The authors of SimSiam emphasize the impact of
        # the `stop_gradient` operator in the paper as it
        # has an important role in the overall optimization.
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        # Negative cosine similarity (minimizing this is
        # equivalent to maximizing the similarity).
        return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

    def compute_InfoNCE_loss(self, p1, p2, temperature=0.2):
        # Negative cosine similarity between positive examples
        # https://medium.com/geekculture/contrastive-learning-without-negative-pairs-28769bdd8410

        p1 = tf.math.l2_normalize(p1, axis=1) / temperature
        p2 = tf.math.l2_normalize(p2, axis=1) / temperature
        loss = tf.reduce_sum((p1 * p2), axis=1) # N

        # Calculate cosine similarity of all original data in the batch
        cos_sim = tf.linalg.matmul(p1,p2,transpose_b=True)
        # Mask out cosine similarity to itself
        mask = tf.eye(cos_sim.shape[0], dtype=tf.dtypes.bool)
        cos_sim = tf.where(mask, -9e15, cos_sim)
        
        # InfoNCE loss
        loss = -loss + tf.reduce_logsumexp(cos_sim, axis=1)
        loss = tf.reduce_mean(loss)

        return loss

    def train_step(self, data):
        # Unpack the data - (ts_data,link_data)
        ds_one, ds_two, indices, links = data
        print(ds_one)
        print(ds_two)
        print(indices)
        print(links)

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)

            if self.loss_fn == 'SimSiam':
                # Note that here we are enforcing the network to match
                # the representations of two differently augmented batches
                # of data.
                loss = self.compute_SimSiam_loss(p1, z2) / 2 + self.compute_SimSiam_loss(p2, z1) / 2
            elif self.loss_fn == 'InfoNCE':
                loss = self.compute_InfoNCE_loss(p1,p2,temperature=self.args['temperature'])
            # link prediction loss
            if links is not None:
                emb1,emb2,y_link = prepare_link_data(p1,indices,links,self.args['batch_size'])
                y_link_pred = self.link_predictor((emb1,emb2))
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                bce_loss = bce(y_link,y_link_pred)
                loss = loss + bce_loss
                tf.print("bce_loss: ", bce_loss)

        # Compute gradients and update the parameters.
        if links is not None:
            learnable_params = (
                self.encoder.trainable_variables + self.predictor.trainable_variables + self.link_predictor.trainable_variables
            )
        else:
            learnable_params = (
                self.encoder.trainable_variables + self.predictor.trainable_variables
            )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
    