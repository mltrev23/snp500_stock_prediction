import yfinance as yf
import ta
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add, GlobalAveragePooling1D
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam

TRAIN_DATA_RATIO = 0.8
VALIDATION_DATA_RATIO = 0.2

NUMBER_OF_SERIES_FOR_PREDICTION = 24

# Download the S&P 500 Data
gspc_data = yf.download('^GSPC', interval='5m', period='1mo')

# Extract the model
feature = pd.DataFrame(index = gspc_data.index)

feature['SMA'] = ta.trend.sma_indicator(gspc_data['Close'], window=14)
feature['MACD'] = ta.trend.macd(gspc_data['Close'])
feature['RSI'] = ta.momentum.rsi(gspc_data['Close'])
feature['Close'] = gspc_data['Close']

gspc_data['SMA'] = feature['SMA']
gspc_data['MACD'] = feature['MACD']
gspc_data['RSI'] = feature['RSI']

# Normalize Feature data that can be the input of the model
mean = {}
std = {}

for key in feature.keys():
    mean[key] = feature[key].mean()
    std[key] = feature[key].std()
    
    feature[key] = (feature[key] - mean[key]) / std[key]

# Split train data and validation data and test data
feature = feature.dropna()
gspc_data = gspc_data.dropna()
train_data_size = int(len(feature) * TRAIN_DATA_RATIO)

train = feature[:train_data_size]
test = feature[train_data_size:]

# Creating dataset for model training
def create_dataset(dataset, number_of_series_for_prediction = 24):
    X_data, y_data = [], []
    
    data_np = np.array(dataset)
    print(data_np.shape)
    
    for i in range(len(data_np) - number_of_series_for_prediction - 1):
        X_data.append(data_np[i : i + number_of_series_for_prediction])
        y_data.append(data_np[i + number_of_series_for_prediction, -1:])
    
    return np.array(X_data), np.array(y_data)

X_train, y_train = create_dataset(train, NUMBER_OF_SERIES_FOR_PREDICTION)
X_test, y_test = create_dataset(test, NUMBER_OF_SERIES_FOR_PREDICTION)

print(f'Dimension of X_train is {X_train.shape}')
print(f'Dimension of y_train is {y_train.shape}')
print(f'Dimension of X_test is {X_test.shape}')
print(f'Dimension of y_test is {y_test.shape}')
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

def build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim, dropout = 0.1):
    inputs = Input(input_shape)
    x = Dense(model_dim)(inputs)
    
    for _ in range(num_layers):
        x = transformer_block(x, model_dim, num_heads, ff_dim, dropout)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    return model

# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

print(f'Dimension of X_train is {X_train}')
print(f'Dimension of y_train is {y_train}')
# print(f'Dimension of X_test is {X_test.shape}')
# print(f'Dimension of y_test is {y_test.shape}')

input_shape = (X_train.shape[1], X_train.shape[2])
model_dim = 64
num_heads = 8
num_layers = 6
ff_dim = 128
output_dim = 1

model = build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim)
print(model.summary())
model.compile(optimizer = Adam(), loss = MeanAbsoluteError(), metrics = [MeanAbsoluteError()])

# Train model
num_epochs = 50
batch_size = 64

model.fit(X_train, y_train, validation_split = VALIDATION_DATA_RATIO, epochs = num_epochs, batch_size = batch_size)

# Evaluate model
loss, mse = model.evaluate(X_test, y_test)
print(f'Test result: Loss {loss}, MSE {mse}')

# Make Predictions
predictions = model.predict(X_test)
predictions = predictions * std['Close'] + mean['Close']

predictions = predictions.flatten()
max_diff = 0
for org, pred in zip(gspc_data.Close[:train_data_size], predictions):
    diff = np.abs(org - pred)
    if max_diff < diff: max_diff = diff
    print(f'Truth: {org}, Prediction: {pred} ----> Diff: {diff}')

print(f'Max Diff: {max_diff}')

gspc_dir = np.where(gspc_data[:train_data_size].shift(-1) > gspc_data[:train_data_size], 1, 0)
pred_dir = np.where(predictions.shift(-1) > predictions, 1, 0)
dir_acc = np.mean(gspc_dir == pred_dir)

print(f'Directioin accuracy: {dir_acc}')