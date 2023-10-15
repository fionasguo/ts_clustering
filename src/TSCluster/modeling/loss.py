import tensorflow as tf


def compute_siam_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

#################################################################
# import numpy as np
# import tensorflow.keras.backend as K
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# def get_res(y_true, y_pred):
#     precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
#     pr_auc = auc(recall, precision)
#     minrp = np.minimum(precision, recall).max()
#     roc_auc = roc_auc_score(y_true, y_pred)
#     return [roc_auc, pr_auc, minrp]
    
# class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=train_op)
# def mortality_loss(y_true, y_pred):
#     sample_weights = (1-y_true)*class_weights[0] + y_true*class_weights[1]
#     bce = K.binary_crossentropy(y_true, y_pred)
#     return K.mean(sample_weights*bce, axis=-1)

# # var_weights = np.sum(fore_train_op[:, V:], axis=0)
# # var_weights[var_weights==0] = var_weights.max()
# # var_weights = var_weights.max()/var_weights
# # var_weights = var_weights.reshape((1, V))
# def forecast_loss(y_true, y_pred):
#     return K.sum(y_true[:,V:]*(y_true[:,:V]-y_pred)**2, axis=-1)

# def get_min_loss(weight):
#     def min_loss(y_true, y_pred):
#         return weight*y_pred
#     return min_loss