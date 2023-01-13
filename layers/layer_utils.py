import tensorflow as tf
from tensorflow.keras import layers, regularizers

def get_dense_layer(output_size,l2_penalty,name):
    return layers.Dense(
        output_size,
        activation="elu",
        kernel_initializer='he_normal',
        kernel_regularizer = regularizers.L2(l2_penalty),
        name=name)

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

#todo - make this work for sparse
#Note that Creating custom layers is better for deserialization
def normalize_adjacency_matrix(adjacency_matrix,use_symmetric_mean, adj_is_sparse):
    if adj_is_sparse:
        degree = layers.Lambda(
            lambda x : tf.sparse.reduce_sum(x, axis=2, keepdims=True, output_is_sparse=True),
            name = "GetDegreeSparse"
        )(adjacency_matrix)

    elif use_symmetric_mean:
        degree = layers.Lambda(
            lambda x : tf.sqrt(
                tf.matmul(
                    tf.math.reduce_sum(x, axis=2, keepdims=True),
                    tf.math.reduce_sum(x, axis=1, keepdims=True)
                )),
            name='GetDegreeSymmtric'
        )(adjacency_matrix)

    else:
        degree = layers.Lambda(
            lambda x : tf.math.reduce_sum(x, axis=2, keepdims=True),
            name='GetDegree'
        )

    return layers.Lambda(
        lambda x: tf.math.divide_no_nan(x[0], x[1]),
        name='Normalize'
    )([adjacency_matrix, degree])
