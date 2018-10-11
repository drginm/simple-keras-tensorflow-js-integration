#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.model_selection import train_test_split

from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

import common

print(tf.__version__)
print(device_lib.list_local_devices())

#%% Load dataset
CSV_PATH = "./dataset/dataset.csv"

df = pd.read_csv(CSV_PATH, index_col=False)

print(df.head())
print(df.columns)

X = df[common.X_colum_names]
Y = df[common.Y_colum_names]

print(X.head(), Y.head())

#%% Split the dataset into different groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print('Training set size: ', len(X_train))
print(X_train.head(), y_train.head())

print('Validation set size: ', len(X_test))
print(X_test.head(), y_test.head())

#%% Create the model
def build_model(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(x_size, input_shape=(x_size,)))

    t_model.compile(loss='mean_squared_error',
        optimizer='sgd',
        metrics=[metrics.mae])
    return(t_model)

print(X_train.shape[1], y_train.shape[1])

model = build_model(X_train.shape[1], y_train.shape[1])
model.summary()

#%% Configure the training
epochs = 50
batch_size = 32

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    ModelCheckpoint(common.model_checkpoint_file_name,
                    monitor='val_mean_absolute_error',
                    save_best_only=True,
                    verbose=0),
    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=2)
]

#%% Train the model
history = model.fit(X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    validation_data=(X_test, y_test),
    callbacks=keras_callbacks)

model.save(common.model_file_name)

train_score = model.evaluate(X_train, y_train, verbose=2)
valid_score = model.evaluate(X_test, y_test, verbose=2)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

#%% See training results
plt.style.use('ggplot')

def plot_history(history, x_size, y_size):
    print(history.keys())  
    # Prepare plotting
    plt.rcParams["figure.figsize"] = [x_size, y_size]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarize history for MAE
    plt.subplot(211)
    plt.plot(history['mean_absolute_error'])
    plt.plot(history['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot the results
    plt.draw()
    plt.show()

plot_history(history.history, x_size=8, y_size=12)
