import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.engine.keras_tensor import SparseKerasTensor
from keras.engine import data_adapter

def initialize_gcn_convolution_fn():
    global sparse_degree_layer
    sparse_degree_layer = layers.Lambda(
        lambda x : tf.sparse.reduce_sum(x, axis=2, keepdims=True, output_is_sparse=True),
        name = "GetDegreeSparse"
    )

    global symmetric_degree_layer
    symmetric_degree_layer = layers.Lambda(
        lambda x : tf.sqrt(
            tf.matmul(
                tf.math.reduce_sum(x, axis=2, keepdims=True),
                tf.math.reduce_sum(x, axis=1, keepdims=True)
            )),
        name='GetDegreeSymmtric'
    )

    global degree_layer
    degree_layer = layers.Lambda(
        lambda x : tf.math.reduce_sum(x, axis=2, keepdims=True),
        name='GetDegree'
    )

    global normalize
    normalize = layers.Lambda(
        lambda x: tf.math.divide_no_nan(x[0], x[1]),
        name='Normalize'
    )

    global propogate_layer
    propogate_layer = layers.Lambda(
        lambda x : tf.matmul(x[0],x[1]),
        name="Propogate"
    )


#does having the padding matter - since we don't have bias these should stay zero
#does this have an output forward layer?  don't think it does
def multihead_self_attention_fn(node_features,mask,params):
    attention_heads = params.get('attention_heads',4)
    l2_penalty = params.get('l2_penalty', 0)
    hidden_size = params.get('hidden_size',64)

    return tf.keras.layers.MultiHeadAttention(
        attention_heads,
        hidden_size // attention_heads,
        dropout=0.0,
        kernel_regularizer = regularizers.L2(l2_penalty))(
            node_features,
            node_features,
            attention_mask = mask
        )

def attention_combination_fn(node_features,row_mask,params):
    square_mask = get_square_mask(row_mask)
    node_features = multihead_self_attention_fn(node_features,square_mask,params)
    return masked_mean(node_features,row_mask)

#loop this
def transformer_attention_combination_fn(node_features,row_mask,params):
    l2_penalty = params.get('l2_penalty',0)
    hidden_size = params.get('hidden_size',64)

    square_mask = get_square_mask(row_mask)

    attention_output = multihead_self_attention_fn(node_features,square_mask,params)
    node_features = node_features + attention_output
    node_features = tf.keras.layers.LayerNormalization()(node_features)

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
    return masked_mean(node_features,row_mask)

#works in this case.  for general case we may need to scale the abs value
#of the max and then min this with 1.
def get_row_mask(x):
    return tf.reduce_max(x,axis=-1,keepdims=True)

def get_square_mask(row_mask):
    return tf.matmul(row_mask, tf.transpose(row_mask,[0,2,1]))

def masked_mean(x,row_mask):
    embeddings_sum = tf.reduce_sum(x, axis=1,keepdims=True)
    masked_embeddings_sum = tf.multiply(embeddings_sum, row_mask)
    return tf.reduce_mean(masked_embeddings_sum,axis=-1)

def get_dense_layer(output_size,params,name):
    l2_penalty=params.get('l2_penalty',0)

    return layers.Dense(
        output_size,
        activation="elu",
        kernel_initializer='he_normal',
        kernel_regularizer = regularizers.L2(l2_penalty),
        name=name)

def mean_combination_fn(node_features, row_mask, params):
    adj_is_ragged = params.get('adj_is_ragged', False)

    if adj_is_ragged:
        return layers.Lambda(lambda x : tf.math.reduce_mean(x,axis=1), name="CombineByMean")(node_features)
    else:
        return layers.Lambda(
            lambda x_mask: masked_mean(x_mask[0],x_mask[1]),
            name="CombineByMean"
        )([node_features,row_mask])


#note that this applies the sigmoid - from_logits should be false in the loss layer
def binary_output_fn(output_size,params):
    l2_penalty = params.get('l2_penalty',0)
    return layers.Dense(
        output_size,
        activation="sigmoid",
        kernel_regularizer = regularizers.L2(l2_penalty),
        name="BinaryOutput")
        #bias_constraint=keras.constraints.MinMaxNorm(min_value=-1, max_value=1.0))

def softmax_output_fn(output_size):
    return layers.Dense(output_size, activation="softmax", name="MulticlassOutput")


#todo - make this work for sparse
#Note that Creating custom layers is better for deserialization
def normalize_adjacency_matrix(adjacency_matrix,use_symmetric_mean, adj_is_sparse):
    if adj_is_sparse:
        degree = sparse_degree_layer(adjacency_matrix)

    elif use_symmetric_mean:
        degree = symmetric_degree_layer(adjacency_matrix)

    else:
        degree = degree_layer(adjacency_matrix)

    return normalize([adjacency_matrix, degree])

#the use_sparse options are commented out b/c they don't seem to work for 3-D matrices?  check on this
def gcn_convolution_fn(dense_layer, node_features, adjacency_matrix,row_mask,params):
    adj_is_sparse = params.get('adj_is_sparse', False)
    use_symmetric_mean = params.get('use_symmetric_mean', ~adj_is_sparse)
    #tf_matmul_fts_is_sparse = params.get('tf_matmul_fts_is_sparse', False)
    #tf_matmul_adj_is_sparse = params.get('tf_matmul_adj_is_sparse', False)

    #Transform
    transformed_node_features = dense_layer(node_features)

    #Zero out empty layers - would be better to not add the bias in the first place - or can i use a mask layer?
    # but this works for now (technically would fail where row sums to zero)
    transformed_node_features = tf.multiply(transformed_node_features,row_mask)
    #transformed_node_features = mask_layer([transformed_node_features, mask])


    #Propogate
    normalized_adjacency_matrix = normalize_adjacency_matrix(
        adjacency_matrix,
        use_symmetric_mean,
        adj_is_sparse
    )

    # Creating a custom layer is better for deserialization
    propogated = propogate_layer(
            [normalized_adjacency_matrix,transformed_node_features]
    )
    return propogated

def GCN (num_atoms,
         feature_size,
         output_fn,
         params):

  adj_is_ragged= params.get('adj_is_ragged', False)
  adj_is_sparse= params.get('adj_is_sparse', False)
  convolution_steps = params.get('convolution_steps', 1)
  convolution_fn = params.get('convolution_fn', gcn_convolution_fn)
  combination_fn = params.get('combination_fn', mean_combination_fn)
  hidden_size = params.get('hidden_size', 64)

  input_node_features = keras.Input(shape=[num_atoms, feature_size], name="features")

  if adj_is_ragged:
      adjacency_matrix = keras.Input(shape=[None,num_atoms], name="adj", sparse=adj_is_sparse)
  else:
      adjacency_matrix = keras.Input(shape=[num_atoms,num_atoms], name="adj", sparse=adj_is_sparse)

  row_mask = get_row_mask(input_node_features)

  initialize_gcn_convolution_fn()
  node_features = convolution_fn(
      get_dense_layer(hidden_size,params,"Transform0"),
      input_node_features,
      adjacency_matrix,
      row_mask,
      params
  )

  for step in range(1,convolution_steps):
      dense_layer = get_dense_layer(hidden_size,params,"Transform" + str(step))
      node_features = convolution_fn(dense_layer,node_features,adjacency_matrix,row_mask,params)

  #combine from multiple atom vectors to one molecule vector
  molecule_vector = combination_fn(node_features,row_mask,params)

  #output
  if output_fn is not None:
      output = output_fn(molecule_vector)
  else:
      output = molecule_vector

  return keras.Model(inputs=[input_node_features,adjacency_matrix], outputs=output, name="GCN")

def ATOM_MULTICLASS_GCN(num_atoms,
                        feature_size,
                        output_size,
                        **kwargs):
  return GCN(num_atoms, feature_size, softmax_output_fn(output_size), kwargs)

#kwargs
#hidden_size (default 64)
#use_symmetric_mean (defaults to true when not sparse)
#convolution_steps (default 1)
#adj_is_ragged (default false)
#adj_is_sparse (default false)
#l2_penalty (default 0)
#tf_matmul_fts_is_sparse - doesn't work
#tf_matmul_adj_is_sparse - doesn't work

def ATOM_GCN(num_atoms,
             feature_size,
             output_size,
             **kwargs):
  return GCN(num_atoms, feature_size, binary_output_fn(output_size,kwargs), kwargs)
