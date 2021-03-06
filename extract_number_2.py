import numpy as np
import pandas as pd
import os
import pickle
import idx2numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def generate_df(data_path, labels_path, inverse_pic=True, noise=True):
    data_raw = idx2numpy.convert_from_file(data_path)
    labels = idx2numpy.convert_from_file(labels_path)
    if inverse_pic:
        # put this in format white background black number
        data_raw = np.uint8(abs(np.int32(data_raw) - 255))
    if noise:
        map = lambda x: np.random.randint(140, 160) if x > 200 else np.random.randint(90, 110)
        vfunc = np.vectorize(map)
        data_raw = vfunc(data_raw)
    df = pd.DataFrame(data_raw.reshape((data_raw.shape[0], 784)))
    df['value'] = labels
    return df


def process_df(mnist_df, sudo_df):
    mnist_df = (mnist_df.loc[mnist_df['value'] > 0]).astype(np.int32)
    mnist_df.columns = mnist_df.columns.astype(str)
    sudo_df = (sudo_df.drop(columns=['is_blank'])).astype(np.int32)
    return pd.concat([mnist_df, sudo_df])


# train_df_mnist = generate_df('mnist_dataset/train-images-idx3-ubyte', 'mnist_dataset/train-labels-idx1-ubyte')
# test_df_mnist = generate_df('mnist_dataset/t10k-images-idx3-ubyte', 'mnist_dataset/t10k-labels-idx1-ubyte')
# train_df_sudo = pd.read_csv('doku_ds/sudoku_train_28x28.csv')
# test_df_sudo = pd.read_csv('doku_ds/sudoku_test_28x28.csv')
#
# train_df = process_df(train_df_mnist, train_df_sudo)
# test_df = process_df(test_df_mnist, test_df_sudo)
# save_train = train_df.astype(np.uint8)
# save_test = test_df.astype(np.uint8)
# save_train.to_csv("doku_ds/sudo_mnist_noised_train.csv", index=False)
# save_test.to_csv("doku_ds/sudo_mnist_noised_test.csv", index=False)

'''train = pd.read_csv('doku_ds/sudo_mnist_noised_train.csv').astype(np.int32)
test = pd.read_csv('doku_ds/sudo_mnist_noised_test.csv').astype(np.int32)

df_train = train.drop(columns=['value'])  # / 255
df_test = test.drop(columns=['value'])  # / 255
label_train = train['value']
label_test = test['value']

input_shape = [df_train.shape[1]]  # shit need to be a list
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    # layers.BatchNormalization(),
    # layers.Dropout(.3),
    layers.Dense(256, activation='relu'),
    # layers.BatchNormalization(),
    # layers.Dropout(.3),
    layers.Dense(128, activation='relu'),
    # layers.BatchNormalization(),
    # layers.Dropout(.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

X_train, X_valid, y_train, y_valid = train_test_split(df_train, label_train, test_size=.15, random_state=42)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=300,
    callbacks=[early_stopping],
)


model.save('save_model\digit_recognizer_noised')'''
load_model = tf.keras.models.load_model('save_model\digit_recognizer_noised')

# load_model.summary()

# test = pd.read_csv('doku_ds/sudo_mnist_noised_test.csv').astype(np.int32)
# df_test = test.drop(columns=['value'])
# label_test = test['value']
# load_model.evaluate(df_test, label_test)


import doc_scan2
import Solver
board = doc_scan2.generate_board_df('test_pic/test_board_2.jpg').astype(np.float32)  # / 255
board_raw = load_model(board.values)
board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
print(board_clean)
Solver.run_solver(board_clean)