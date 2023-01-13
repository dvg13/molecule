import layer_utils as utils
import tensorflow as tf

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
    normalized_adjacency_matrix = utils.normalize_adjacency_matrix(
        adjacency_matrix,
        use_symmetric_mean,
        adj_is_sparse
    )

    # Creating a custom layer is better for deserialization
    propogated = layers.Lambda(
            lambda x : tf.matmul(x[0],x[1]),
            name="Propogate"
    )[
        normalized_adjacency_matrix,
        transformed_node_features
    ]

    return propogated
