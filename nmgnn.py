#Todo
#support batches
#support metrics besides loss

import tensorflow as tf
from tensorflow import keras
import numpy as np

def normalize_adjacency_matrix(adjacency_matrix,use_symmetric_mean):
    degree = tf.math.reduce_sum(adjacency_matrix, axis=2, keepdims=True)

    if use_symmetric_mean:
        sqrt_degree = tf.math.sqrt(degree)
        degree = tf.matmul(sqrt_degree, tf.transpose(sqrt_degree, perm=[0,2,1]))

    return tf.math.divide_no_nan(adjacency_matrix, degree)

def extract_data(X,y=None,idx=0):
    #data generator
    if isinstance(X, keras.utils.Sequence):
        [features,adj], label = X.__getitem__(idx)
        return features, adj, label
    #numpy
    [features, adj] = X[idx]
    if y is not None:
        label = y[idx]
    return features, adj if y is None else features, adj, label

def initialize_gcn_convolution_fn(self, params):
    self.conv_layer_1 = keras.layers.Dense(params["hidden_size"], activation="relu", name="Transform")

def run_gcn_convolution_fn(self, node_features, adjacency_matrix, params):
    transformed_node_features = self.conv_layer_1(node_features)

    #Propogate
    use_symmetric_mean = params["user_symmetric_mean"] if "use_symmetric_mean" in params else True
    normalized_adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix, use_symmetric_mean)

    return tf.matmul(normalized_adjacency_matrix, transformed_node_features)

gcn_convolution_fn = {"init": initialize_gcn_convolution_fn, "run": run_gcn_convolution_fn}

def initialize_binary_output_fn(self, params):
    self.output_layer_1 = keras.layers.Dense(params["output_size"], name="BinaryOutput")

def run_binary_output_fn(self, input):
    return self.output_layer_1(input)

binary_output_fn = {"init": initialize_binary_output_fn, "run": run_binary_output_fn}

#come back and do for batches
class ATOM_GCN():
    def __init__(self,
                 output_size=1,
                 hidden_size=64,
                 convolution_fn=gcn_convolution_fn,
                 combination_fn=lambda x: tf.math.reduce_mean(x,axis=1),
                 output_fn=binary_output_fn):
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.convolution_fn = convolution_fn
        self.combination_fn = combination_fn
        self.output_fn = output_fn

    def compile(self,
                optimizer=tf.keras.optimizers.Adam(.1),
                loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        #not_strictly necessary
        self.params = self.__dict__

        self.convolution_fn["init"](self, self.params)
        self.output_fn["init"](self, self.params)

    def forward(self, node_features, adjacency_matrix):
        convolved_node_features = self.convolution_fn["run"](self, node_features,adjacency_matrix, self.params)

        graph_vector = self.combination_fn(convolved_node_features)

        #output
        if self.output_fn is not None:
            logits = self.output_fn["run"](self, graph_vector)
        else:
            logits = graph_vector

        return logits

    #expand to get other metrics
    def evaluate(self, X, y=None, steps=None):
        loss = 0
        steps = len(X) if steps is None else steps
        for idx in range(steps):
            node_features, adjacency_matrix, labels = extract_data(X,y,idx)
            logits = self.forward(node_features, adjacency_matrix)
            loss += (self.loss_fn(labels,logits).numpy() / len(X))
        return loss

    def fit(self, X, y=None, epochs=1, shuffle=True, valid_X=None, valid_y=None):
        indices = np.arange(len(X))

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(indices)

            print("Epoch {}/{}".format(epoch+1,epochs))
            prog_bar = tf.keras.utils.Progbar(len(X), stateful_metrics='valid_loss')

            for idx in range(len(indices)):
                node_features, adjacency_matrix, labels = extract_data(X,y,indices[idx])

                with tf.GradientTape() as t:
                    logits = self.forward(node_features,adjacency_matrix)
                    loss = self.loss_fn(labels,logits)
                    variables = t.watched_variables()
                    grads = t.gradient(loss, variables)
                    self.optimizer.apply_gradients(zip(grads, variables))

                    prog_bar.update(idx, values=[('train_loss', loss.numpy())])

            if valid_X is not None:
                prog_bar.update(len(X), values=[('valid_loss', self.evaluate(valid_X, valid_y))])

    def predict(self, X, steps=None, return_logits=True):
        steps = len(X) if steps is None else steps
        prediction = tf.stack(
            [self.forward(*extract_data(X,None,idx)[0:2]) for idx in range(steps)]
        )

        if not return_logits:
            prediction = tf.nn.sigmoid(prediction)

        return prediction