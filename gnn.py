import numpy as np
import tensorflow as tf
import layers.layer_utils as utils

from layers.combination_layers import mean_combination_fn
from layers.convolution_layers import gcn_convolution_fn
from layers.output_layers import binary_output_fn
from tensorflow import keras

def GCN (num_atoms,
         feature_size,
         params):

  adj_is_ragged= params.get('adj_is_ragged', False)
  adj_is_sparse= params.get('adj_is_sparse', False)
  convolution_steps = params.get('convolution_steps', 1)
  convolution_fn = params.get('convolution_fn', gcn_convolution_fn)
  combination_fn = params.get('combination_fn', mean_combination_fn)
  output_fn = params.get('output_fn', binary_output_fn)
  hidden_size = params.get('hidden_size', 64)

  input_node_features = keras.Input(shape=[num_atoms, feature_size], name="features")

  if adj_is_ragged:
      adjacency_matrix = keras.Input(shape=[None,num_atoms], name="adj", sparse=adj_is_sparse)
  else:
      adjacency_matrix = keras.Input(shape=[num_atoms,num_atoms], name="adj", sparse=adj_is_sparse)

  row_mask = utils.get_row_mask(input_node_features)

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

#kwargs
#hidden_size (default 64)
#use_symmetric_mean (defaults to true when not sparse)
#convolution_steps (default 1)
#adj_is_ragged (default false)
#adj_is_sparse (default false)
#l2_penalty (default 0)
#convolution_fn = default gcn_convolution_fn
#combination_fn = default mean_combination_fn
#output_fn = default binary_output_fn
#tf_matmul_fts_is_sparse - doesn't work
#tf_matmul_adj_is_sparse - doesn't work

def ATOM_GCN(num_atoms,
             feature_size,
             output_size,
             **kwargs):
  return GCN(num_atoms, feature_size, binary_output_fn(output_size,kwargs), kwargs)

def ATOM_MULTICLASS_GCN(num_atoms,
                        feature_size,
                        output_size,
                        **kwargs):
  return GCN(num_atoms, feature_size, kwargs)
