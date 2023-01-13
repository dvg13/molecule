import numpy as np
import tensorflow as tf
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
