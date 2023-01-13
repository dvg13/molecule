from tensorflow.keras import layers, regularizers
from .layers import layer_utils as utils

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
