#/usr/bin/python3

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepchem.feat.mol_graphs import ConvMol
import dataloader_utils as utils

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
    conv_mol_fn = lambda i: utils.unpack_conv_mol(self.dataset, i, self.adj_is_sparse, self.adj_is_ragged, self.max_atoms)

    #extract features
    feature_fn = lambda i: conv_mol_fn(i)[0]
    if self.max_atoms is not None:
        feature_fn = lambda i: utils.pad(conv_mol_fn(i)[0], [0], [self.max_atoms])

    #extract adj
    adj_fn = lambda i: conv_mol_fn(i)[1]

    #pad if necessary
    if not self.adj_is_sparse and self.max_atoms is not None:
        if self.adj_is_ragged:
            adj_fn = lambda i: utils.pad(conv_mol_fn(i)[1], [1], [self.max_atoms])
        else:
            adj_fn = lambda i: utils.pad(conv_mol_fn(i)[1], [0,1], [self.max_atoms, self.max_atoms])

    X = tf.stack([feature_fn(i) for i in batch_range])
    adj = (tf.sparse.concat(0, [tf.sparse.expand_dims(adj_fn(i),0) for i in batch_range])) if self.adj_is_sparse else (
        utils.create_ragged_tensor(adj_fn, batch_range)) if self.adj_is_ragged else (
        tf.stack([adj_fn(i) for i in batch_range]))
    y = tf.stack([conv_mol_fn(i)[2] for i in batch_range])

    return [X,adj],y

  def get_in_memory_vectors(self):
    self.batch_size = self.__len__()
    [features,adj],labels = self.__getitem__(0)
    return features, adj, labels
