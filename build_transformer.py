import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add, GlobalAveragePooling1D

# Building a model
def transformer_block(inputs, model_dim, num_heads, ff_dim, dropout = 0.1):
    # Multi-head attention layer
    attention_output = MultiHeadAttention(num_heads = num_heads, key_dim = model_dim)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    output1 = LayerNormalization(epsilon = 1e-6)(inputs + attention_output)
    
    # Feed-forward layer
    ff_output = Dense(ff_dim, activation = 'relu')(output1)
    ff_output = Dense(model_dim)(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    output2 = LayerNormalization(epsilon = 1e-6)(output1 + ff_output)
    
    return output2

def positional_encoding(max_position, model_dim):
    angle_rads = np.arange(max_position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(model_dim)[np.newaxis, :] // 2)) / np.float32(model_dim))
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim, dropout = 0.1):
    inputs = Input(input_shape)
    x = Dense(model_dim)(inputs)
    #position_encoding = positional_encoding(input_shape[0], model_dim)
    #x = x + position_encoding
    
    for _ in range(num_layers):
        x = transformer_block(x, model_dim, num_heads, ff_dim, dropout)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    return model
