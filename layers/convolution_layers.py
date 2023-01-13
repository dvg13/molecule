import tensorflow as tf
from . import layer_utils as utils
from tensorflow.keras import layers,regularizers

#refactor this as a layer so that we don't initialize in the function
#the use_sparse options are commented out b/c they don't seem to work for 3-D matrices?  check on this
def gcn_convolution_fn(
        node_features,
        adjacency_matrix,
        row_mask,
        num_convolutions,
        hidden_size,
        params):
    l2_penalty = params.get('l2_penalty', 0)
    adj_is_sparse = params.get('adj_is_sparse', False)
    use_symmetric_mean = params.get('use_symmetric_mean', ~adj_is_sparse)
    #tf_matmul_fts_is_sparse = params.get('tf_matmul_fts_is_sparse', False)
    #tf_matmul_adj_is_sparse = params.get('tf_matmul_adj_is_sparse', False)

    #initialize layers
    propogate_layer = layers.Lambda(
        lambda x : tf.matmul(x[0],x[1]),
        name="Propogate"
    )

    normalized_adjacency_matrix = utils.normalize_adjacency_matrix(
        adjacency_matrix,
        use_symmetric_mean,
        adj_is_sparse)

    zero_out_layer = layers.Lambda(
        lambda x: tf.multiply(x[0],x[1]),
        name="ZeroOut"
    )

    #Transform
    for convolution in range(num_convolutions):
        node_features = layers.Dense(
            hidden_size,
            activation="elu",
            kernel_initializer='he_normal',
            kernel_regularizer = regularizers.L2(l2_penalty),
            name="Transform" + str(convolution)) (node_features)

        #Zero out empty layers - would be better to not add the bias in the first place - or can i use a mask layer?
        # but this works for now (technically would fail where row sums to zero)
        node_features = zero_out_layer([node_features, row_mask])
        #transformed_node_features = mask_layer([transformed_node_features, mask])

        # Creating a custom layer is better for deserialization
        node_features = propogate_layer([
            normalized_adjacency_matrix,
            node_features
        ])

    return node_features
