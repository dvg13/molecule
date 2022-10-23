import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.engine.keras_tensor import SparseKerasTensor
from keras.engine import data_adapter


def padded_mean(x):
    row_sums = tf.reduce_sum(x, axis=2)
    non_zero_rows = tf.expand_dims(tf.math.count_nonzero(row_sums, axis=1, dtype=tf.float32), axis=1)
    embeddings_sum = tf.reduce_sum(x, axis=1)
    return tf.math.divide_no_nan(embeddings_sum, non_zero_rows)

def mean_combination_fn(node_features, params):
    adj_is_ragged = params.get('adj_is_ragged', False)

    if adj_is_ragged:
        return layers.Lambda(lambda x : tf.math.reduce_mean(x,axis=1), name="CombineByMean")(node_features)
    else:
        return layers.Lambda(lambda x : padded_mean(x), name="CombineByMean")(node_features)


#note that this applies the sigmoid - from_logits should be false in the loss layer
def binary_output_fn(output_size,params):
    l2_penalty = params.get('l2_penalty',0)
    return layers.Dense(
        output_size,
        activation="sigmoid",
        kernel_regularizer(regularizers.L2(l2_penalty))
        name="BinaryOutput")
        #bias_constraint=keras.constraints.MinMaxNorm(min_value=-1, max_value=1.0))

def softmax_output_fn(output_size):
    return layers.Dense(output_size, activation="softmax", name="MulticlassOutput")


#todo - make this work for sparse
#Note that Creating custom layers is better for deserialization
def normalize_adjacency_matrix(adjacency_matrix,use_symmetric_mean, adj_is_sparse):
    if adj_is_sparse:
        degree = layers.Lambda(
            lambda x : tf.sparse.reduce_sum(x, axis=2, keepdims=True, output_is_sparse=True)
        )(adjacency_matrix)

    elif use_symmetric_mean:
        degree = layers.Lambda(
            lambda x : tf.sqrt(
                tf.matmul(
                    tf.math.reduce_sum(x, axis=2, keepdims=True),
                    tf.math.reduce_sum(x, axis=1, keepdims=True)
                )),
            name='GetSymmtricMean'
        )(adjacency_matrix)

    else:
        degree = layers.Lambda(lambda x : tf.math.reduce_sum(x, axis=2, keepdims=True),
                               name='GetDegree')(adjacency_matrix)

    return layers.Lambda(lambda x: tf.math.divide_no_nan(x[0], x[1]), name='Normalize')([adjacency_matrix, degree])

#the use_sparse options are commented out b/c they don't seem to work for 3-D matrices?  check on this
def gcn_convolution_fn(node_features, adjacency_matrix, params):
    hidden_size = params.get('hidden_size', 64)
    adj_is_sparse = params.get('adj_is_sparse', False)
    use_symmetric_mean = params.get('use_symmetric_mean', ~adj_is_sparse)
    l2_penalty = params.get('l2_penalty', 0)
    #tf_matmul_fts_is_sparse = params.get('tf_matmul_fts_is_sparse', False)
    #tf_matmul_adj_is_sparse = params.get('tf_matmul_adj_is_sparse', False)

    #Transform
    transformed_node_features = layers.Dense(
        hidden_size,
        activation="elu",
        kernel_initializer='he_normal',
        kernel_regularizer(regularizers.L2(l2_penalty)),
        name="Transform")(node_features)

    #Zero out empty layers - would be better to not add the bias in the first place - or can i use a mask layer?
    # but this works for now (technically would fail where row sums to zero)
    mask = tf.expand_dims(tf.cast(tf.math.reduce_sum(node_features, axis=2) != 0, tf.float16), axis=2)
    transformed_node_features = tf.keras.layers.Multiply(name='ZeroOutPadding')([transformed_node_features, mask])


    #Propogate
    normalized_adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix, use_symmetric_mean, adj_is_sparse)

    # Creating a custom layer is better for deserialization
    propogated = layers.Lambda(
        lambda x : tf.matmul(x[0],
                             x[1],
                             #a_is_sparse=tf_matmul_adj_is_sparse,
                             #b_is_sparse=tf_matmul_fts_is_sparse),
                            ), name="Propogate")(
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
  combination_fn = params.get('combination_fn', lambda x, params: mean_combination_fn(x, params))

  input_node_features = keras.Input(shape=[num_atoms, feature_size], name="features")

  if adj_is_ragged:
      adjacency_matrix = keras.Input(shape=[None,num_atoms], name="adj", sparse=adj_is_sparse)
  else:
      adjacency_matrix = keras.Input(shape=[num_atoms,num_atoms], name="adj", sparse=adj_is_sparse)

  for step in range(convolution_steps):
      node_features = convolution_fn(input_node_features, adjacency_matrix, params)

  #combine from multiple atom vectors to one molecule vector
  graph_vector = combination_fn(node_features, params)

  #output
  if output_fn is not None:
      output = output_fn(graph_vector)
  else:
      output = graph_vector

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
  return GCN(num_atoms, feature_size, binary_output_fn(output_size), kwargs)
