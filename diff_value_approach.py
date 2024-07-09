import yfinance as yf
import ta
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
from build_transformer import build_transformer_model
from build_dense_layer import build_model

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
    
    for i in range(len(data_np) - number_of_series_for_prediction):
        X_data.append(data_np[i : i + number_of_series_for_prediction])
        y_data.append([data_np[i + number_of_series_for_prediction, -1] - data_np[i + number_of_series_for_prediction - 1, -1]])
    y_data = np.where(np.array(y_data) > 0, 1, -1)
    
    return np.array(X_data), np.array(y_data)

X_train, y_train = create_dataset(train, NUMBER_OF_SERIES_FOR_PREDICTION)
X_test, y_test = create_dataset(test, NUMBER_OF_SERIES_FOR_PREDICTION)

print(f'Dimension of X_train is {X_train.shape}')
print(f'Dimension of y_train is {y_train.shape}')
print(f'Dimension of X_test is {X_test.shape}')
print(f'Dimension of y_test is {y_test.shape}')

# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

print(f'Dimension of X_train is {X_train}')
print(f'Dimension of y_train is {y_train}')
# print(f'Dimension of X_test is {X_test.shape}')
# print(f'Dimension of y_test is {y_test.shape}')

# input_shape = (X_train.shape[1], X_train.shape[2])
# model_dim = 64
# num_heads = 8
# num_layers = 6
# ff_dim = 128
# output_dim = 1

# model = build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

input_shape = X_train.shape[1]
output_dim = 1
model = build_model(input_shape, output_dim, 128)
print(model.summary())

# Direction Accuracy Metric
def direction_accuracy(y_true, y_pred):
    print(f'y_true {y_true}')
    print(f'y_pred {y_pred}')
    direction_true = tf.sign(y_true[:, 1:] - y_true[:, :-1])
    direction_pred = tf.sign(y_pred[:, 1:] - y_pred[:, :-1])
    correct_directions = tf.equal(direction_true, direction_pred)
    return tf.reduce_mean(tf.cast(correct_directions, tf.float32))
model.compile(optimizer = Adam(), loss = MeanAbsoluteError(), metrics = ['mse'])

# Custorm Learning Rate Schedular
def custom_lr_schedule(epoch, lr):
    warmup_epochs = 10
    warmup_lr = 1e-4
    initial_lr = 1e-3
    decay_rate = 0.4
    decay_step = 10
    
    if epoch < warmup_epochs:
        lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / warmup_epochs)
    else:
        lr = initial_lr * (decay_rate ** ((epoch - warmup_epochs) / decay_step))
    return lr

# Early Stopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20, min_delta = 1e-4, mode = 'min', restore_best_weights = True)

# Model checkpoint
model_checkpoint = ModelCheckpoint(filepath = 'model_checkpoint.keras', save_best_only = True, monitor = 'val_loss', mode = 'min', verbose = 1)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6, verbose = 1)

# Train model
num_epochs = 200
batch_size = 64

lr_scheduler = LearningRateScheduler(custom_lr_schedule)
model.fit(X_train, y_train, validation_split = VALIDATION_DATA_RATIO, epochs = num_epochs, batch_size = batch_size, callbacks = [early_stopping, reduce_lr, model_checkpoint])

# Evaluate model
loss, mse = model.evaluate(X_test, y_test)
print(f'Test result: Loss {loss}, MSE {mse}')  

# test trained data predictions
prediction_train_data = model.predict(X_train)

# Plotting the actual and predicted values
plt.figure(figsize=(20, 14))
plt.plot(y_train, label='Actual values', color='blue')  
plt.plot(prediction_train_data, label='Predicted values', color='red', linestyle='--')
plt.title('Actual vs Predicted values')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('diff_traindata_inference.png')

first = train.Close[NUMBER_OF_SERIES_FOR_PREDICTION - 1]

print(f'first {first}')
prediction_train_data[0] += first
for i in range(1, len(prediction_train_data)):
    prediction_train_data[i][0] += prediction_train_data[i - 1][0]

prediction_train_data = prediction_train_data * std['Close'] + mean['Close']

actual_close_prices = np.array(gspc_data.Close[NUMBER_OF_SERIES_FOR_PREDICTION: train_data_size])

# Plotting the actual and predicted values
plt.figure(figsize=(20, 14))
plt.plot(actual_close_prices, label='Actual Close Prices', color='blue')  
plt.plot(prediction_train_data, label='Predicted Close Prices', color='red', linestyle='--')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('diff_training.png')


# Make Predictions
predictions = model.predict(X_test)
first = test.Close[NUMBER_OF_SERIES_FOR_PREDICTION - 1]

print(f'first {first}')
predictions[0] += first
for i in range(1, len(predictions)):
    predictions[i][0] += predictions[i - 1][0]

predictions = predictions * std['Close'] + mean['Close']
print(predictions.shape)

predictions = predictions.flatten()
print(f'after flatten {predictions.shape}')

actual_close_prices = gspc_data.Close[train_data_size + NUMBER_OF_SERIES_FOR_PREDICTION:]
print(f'actual_close_prices {actual_close_prices.shape}')

max_diff = 0
for org, pred in zip(actual_close_prices, predictions):
    diff = np.abs(org - pred)
    if max_diff < diff: max_diff = diff
    print(f'Truth: {org}, Prediction: {pred} ----> Diff: {diff}')

print(f'Max Diff: {max_diff}')

actual_close_prices = np.array(actual_close_prices)
print(f'actual_close_prices {actual_close_prices.shape}')

gspc_dir = np.where(actual_close_prices[:-1] > actual_close_prices[1:], 1, 0)
pred_dir = np.where(predictions[:-1] > predictions[1:], 1, 0)
dir_acc = np.mean(gspc_dir == pred_dir)

print(f'Direction accuracy: {dir_acc}')

model.save('diff_training.h5')

# Plotting the actual and predicted values
plt.figure(figsize=(20, 14))
plt.plot(actual_close_prices, label='Actual Close Prices', color='blue')
plt.plot(predictions, label='Predicted Close Prices', color='red', linestyle='--')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('diff_test.png')