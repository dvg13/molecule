import tensorflow as tf
from . import layer_utils as utils
from tensorflow.keras import layers, regularizers

#this shouldn't be self attention, make it global with and initialize the
#initial weights
def attention_combination_fn(node_features,row_mask,params):
    square_mask = utils.get_square_mask(row_mask)
    node_features = multihead_self_attention_fn(node_features,square_mask,params)
    return utils.masked_mean(node_features,row_mask)

def mean_combination_fn(node_features, row_mask, params):
    adj_is_ragged = params.get('adj_is_ragged', False)

    if adj_is_ragged:
        return layers.Lambda(lambda x : tf.math.reduce_mean(x,axis=1), name="CombineByMean")(node_features)
    else:
        return layers.Lambda(
            lambda x_mask: utils.masked_mean(x_mask[0],x_mask[1]),
            name="CombineByMean"
        )([node_features,row_mask])

#loop this
def transformer_attention_combination_fn(node_features,row_mask,params):
    l2_penalty = params.get('l2_penalty',0)
    hidden_size = params.get('hidden_size',64)

    square_mask = get_square_mask(row_mask)

    attention_output = multihead_self_attention_fn(node_features,square_mask,params)
    node_features = node_features + attention_output
    node_features = layers.LayerNormalization()(node_features)

    feed_forward_output = layers.Dense(
        hidden_size, #the paper makes this one larger
        activation="relu",
        kernel_initializer='he_normal',
        kernel_regularizer = regularizers.L2(l2_penalty),
        name='feed_forward_1'
    ) (node_features)
    feed_forward_output = layers.Dense(
        hidden_size,
        activation=None,
        kernel_regularizer = regularizers.L2(l2_penalty),
        name='feed_forward_2'
    ) (feed_forward_output)

    node_features = feed_forward_output + node_features
    node_features = tf.keras.layers.LayerNormalization()(node_features)
    return utils.masked_mean(node_features,row_mask)

#does having the padding matter - since we don't have bias these should stay zero
#does this have an output forward layer?  don't think it does
def multihead_self_attention_fn(node_features,mask,params):
    attention_heads = params.get('attention_heads',4)
    l2_penalty = params.get('l2_penalty', 0)
    hidden_size = params.get('hidden_size',64)

    return layers.MultiHeadAttention(
        attention_heads,
        hidden_size // attention_heads,
        dropout=0.0,
        kernel_regularizer = regularizers.L2(l2_penalty))(
            node_features,
            node_features,
            attention_mask = mask
        )
