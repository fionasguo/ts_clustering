import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda, Add, Concatenate, ReLU, BatchNormalization
from tensorflow.keras import regularizers, Model

from .layers import CVE, Attention, Transformer
from .loss import compute_siam_loss


def get_encoder(
        n_feat: int,
        emb_dim: int = 50, 
        max_triplet_len: int = 200, 
        n_transformer_layer: int = 2, 
        n_attn_head: int = 4, 
        dropout: float = 0.2, 
        demo_dim=0
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

    Return:
        embedding vectors
    """
    

    # embed feature dummy, or variable name
    varis = Input(shape=(max_triplet_len,))
    varis_emb = Embedding(n_feat+1, emb_dim)(varis) # output shape (batch size, max_triplet_len_triplets, d)

    # embed timestamps and values
    values = Input(shape=(max_triplet_len,))
    times = Input(shape=(max_triplet_len,))
    cve_units = int(np.sqrt(emb_dim))
    values_emb = CVE(cve_units, emb_dim)(values) # output shape (batch size, max_triplet_len_triplets, d)
    times_emb = CVE(cve_units, emb_dim)(times)

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

    emb = Dense(n_feat)(emb) # for forcasting
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
    model = tf.keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            Input((input_dim,)),
            Dense(
                hid_dim,
                use_bias=False,
                kernel_regularizer=regularizers.l2(weight_decay),
            ),
            ReLU(),
            BatchNormalization(),
            Dense(input_dim),
        ],
        name="predictor",
    )
    return model

# To tune:
# 1. Transformer parameters. (N, h, dropout)
# 2. Normalization

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
            demo_dim=self.args['demo_dim']
        )
        self.predictor = get_predictor(
            input_dim=self.args['n_feat'], ##TODO: figure out the dim
            hid_dim=self.args['embed_dim'], 
            weight_decay=self.args['weight_decay']
        )

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        print(data)
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)

            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = compute_siam_loss(p1, z2) / 2 + compute_siam_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}