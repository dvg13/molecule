#/usr/bin/python3

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepchem.feat.mol_graphs import ConvMol


def adj_list_to_dense_matrix(adj_list):
  matrix = np.eye(len(adj_list), dtype=np.float32)
  for i in range(len(adj_list)):
    for j in adj_list[i]:
      matrix[i][j]=1
  return matrix
  
def adj_list_to_sparse_matrix(adj_list, dims):
    indices = []
    values = []
    for i in range(len(adj_list)):
      for j in adj_list[i]:
        indices.append([i,j])
        values.append(1.0)
    return tf.sparse.SparseTensor(indices, values, dense_shape=dims)
    
def pad(array, axes, new_lengths):
    padded_shape = list(array.shape)
    for axis, length in zip(axes, new_lengths):
        padded_shape[axis] = length
    new_array = np.zeros(padded_shape, dtype=array.dtype)
    old_array_slices = [slice(0,x) for x in array.shape]
    new_array[old_array_slices] = array
    return new_array
    
def unpack_conv_mol(dataset, index, adj_is_sparse, adj_is_ragged, max_atoms):
    num_atoms = len(dataset.X[index].get_adjacency_list())
    if adj_is_sparse:
        if max_atoms is None:
            dims = [num_atoms, num_atoms]
        elif adj_is_ragged:
            dims = [num_atoms, max_atoms] 
        else:
            dims = [max_atoms, max_atoms]
        
        adj_list_to_matrix_fn = lambda x: adj_list_to_sparse_matrix(x, dims)
        
    else:
        adj_list_to_matrix_fn = adj_list_to_dense_matrix
        
    return [
          dataset.X[index].get_atom_features(),
          adj_list_to_matrix_fn(dataset.X[index].get_adjacency_list()),
          dataset.y[index]
    ]
    
def create_ragged_tensor(adj_fn, batch_range):
    values = tf.concat([(adj_fn(i)) for i in batch_range], axis=0)
    row_splits = np.zeros(len(batch_range)+1)
    total_idx = 0
    for idx in range(len(batch_range)):
        total_idx += adj_fn(batch_range[idx]).shape[0]
        row_splits[idx+1] = total_idx
    return tf.RaggedTensor.from_row_splits(values=values, row_splits=row_splits)
        
  
def get_max_atoms(dataset):
  return max(map(ConvMol.get_num_atoms, dataset.X))
  
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class ConvMolGenerator(keras.utils.Sequence):
  def __init__(self, dataset, batch_size=1, max_atoms=None, adj_is_ragged=False, adj_is_sparse=False):
    self.dataset = dataset
    self.examples = len(self.dataset)
    self.max_atoms=max_atoms
    self.adj_is_ragged = adj_is_ragged
    self.adj_is_sparse = adj_is_sparse
    self.batch_size = batch_size
    
    if self.batch_size > 1 and self.max_atoms is None:
        raise Exception('If max_atoms is not specified, batch size must be 1')
    
  def __len__(self):
    return math.ceil(self.examples / self.batch_size)

  def __getitem__(self, idx):
    batch_range = range(idx * self.batch_size, min(self.examples, (idx + 1) * self.batch_size))
    
    conv_mol_fn = lambda i: unpack_conv_mol(self.dataset, i, self.adj_is_sparse, self.adj_is_ragged, self.max_atoms)
    
    #extract features
    feature_fn = lambda i: conv_mol_fn(i)[0]
    if self.max_atoms is not None:
        feature_fn = lambda i: pad(conv_mol_fn(i)[0], [0], [self.max_atoms])
        
    #extract adj 
    adj_fn = lambda i: conv_mol_fn(i)[1]
    #pad if necessary
    if not self.adj_is_sparse and self.max_atoms is not None:
        if self.adj_is_ragged:
            adj_fn = lambda i: pad(conv_mol_fn(i)[1], [1], [self.max_atoms])
        else:
            adj_fn = lambda i: pad(conv_mol_fn(i)[1], [0,1], [self.max_atoms, self.max_atoms])
        
    X = tf.stack([feature_fn(i) for i in batch_range])
    adj = (tf.sparse.concat(0, [tf.sparse.expand_dims(adj_fn(i),0) for i in batch_range])) if self.adj_is_sparse else (
        create_ragged_tensor(adj_fn, batch_range)) if self.adj_is_ragged else (
        tf.stack([adj_fn(i) for i in batch_range]))
    y = tf.stack([conv_mol_fn(i)[2] for i in batch_range])

    return [X,adj],y
    
  def get_in_memory_vectors(self):
    self.batch_size = self.__len__()
    [features,adj],labels = self.__getitem__(0)
    return features, adj, labels
        
