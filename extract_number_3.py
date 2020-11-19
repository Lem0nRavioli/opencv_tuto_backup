import numpy as np
import pandas as pd
import os
import cv2
import pickle
import idx2numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import doc_scan_3 as scanner
import Solver

# pd.options.display.width = None


def generate_df(data_path, labels_path):
    data_raw = idx2numpy.convert_from_file(data_path)
    labels = idx2numpy.convert_from_file(labels_path)
    df = pd.DataFrame(data_raw.reshape((data_raw.shape[0], 784)))
    df['value'] = labels
    df.loc[df['value'] == 0, df.columns[:784]] = 0
    df = df.astype(np.uint8)
    return df


def generate_board_tiles(path):
    img = scanner.extract_sudoku(path)
    s = 30
    tiles = [img[x + 1:x + s - 1, y + 1:y + s - 1] for x in range(0, img.shape[0], s) for y in
             range(0, img.shape[1], s)]
    return tiles



# train_df_mnist = generate_df('mnist_dataset/train-images-idx3-ubyte', 'mnist_dataset/train-labels-idx1-ubyte')
# test_df_mnist = generate_df('mnist_dataset/t10k-images-idx3-ubyte', 'mnist_dataset/t10k-labels-idx1-ubyte')
# train = train_df_mnist.astype(np.int32)
# test = test_df_mnist.astype(np.int32)
#
# df_train = train.drop(columns=['value']) / 255
# df_test = test.drop(columns=['value']) / 255
# label_train = train['value']
# label_test = test['value']

#######################################################################################################################
# TRAIN MODEL


'''

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


model.save('save_model\digit_model_mnist')

'''

load_model = tf.keras.models.load_model('save_model\digit_model_mnist')

# load_model.summary()
#
# load_model.evaluate(df_test, label_test)


test_path = "v2_train/image58.jpg"
path = "test_pic"
image_file = "test_board_1.jpg"
img_path = os.path.join(path, image_file)
tiles = np.array(generate_board_tiles(img_path)).reshape((81, 784)).astype(np.int32) / 255
board_raw = load_model(tiles)
board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
print(board_clean)
Solver.run_solver(board_clean)



# to do :
# * cut the board into 9x9 30x30 pxl k
# * crop them [1:29, 1:29] k
# * train a model with mnist
# * feed the 9x9 [1:29, 1:29] board to the model
# * select argmax and reshape 9x9 answer
# * call solver with board
# * popcorn


# for tile in train_df_mnist.head(2).values:
#     print(str(tile[784]))
#     print(tile[:784].reshape((28, 28)))
#     cv2.imshow("number" + str(tile[784]), tile[:784].reshape((28, 28)))
#
#
# cv2.waitKey(0)