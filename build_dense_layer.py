from keras.models import Model
from keras.layers import Dense, Dropout, Input

def build_model(input_shape, output_shape, hidden_layer_shape):
    inputs = Input(shape=(input_shape,))
    
    x = Dense(hidden_layer_shape, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(hidden_layer_shape, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(hidden_layer_shape * 2, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model