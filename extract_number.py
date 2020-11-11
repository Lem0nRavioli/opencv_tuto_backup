import numpy as np
import pandas as pd
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_evaluate(X, y, model, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=.15)
    mdl = model
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    pickle.dump(mdl, open(filename, 'wb'))
    return accuracy_score(y_test, y_pred)


'''uncomment this part to generate df'''
# import document_scanner
#
# os.makedirs("doku_ds", exist_ok=True)
# df_train = document_scanner.data_frame
# df_test = document_scanner.df_test
# df_train.to_csv("doku_ds/sudoku_train.csv", index=False)
# df_test.to_csv("doku_ds/sudoku_test.csv", index=False)

df_train = pd.read_csv("doku_ds/sudoku_train.csv")
df_test = pd.read_csv("doku_ds/sudoku_test.csv")
# print(df_train.head())

X = df_train.drop(columns=["value", "is_blank"])
y_blank = df_train['is_blank']
y_value = df_train['value']
X_noblank = df_train.loc[df_train['is_blank'] == 0]
y_value_noblank = X_noblank['value']
X_noblank = X_noblank.drop(columns=["value", "is_blank"])

# acc_blank = train_evaluate(X, y_blank, KNeighborsClassifier(n_neighbors=2), "knn_blank.sav")  # .94
# print(acc_blank)
# acc_value = train_evaluate(X, y_value, KNeighborsClassifier(n_neighbors=10), "knn_value.sav")  # .84
# print(acc_value)
# acc_value_noblank = train_evaluate(X_noblank, y_value_noblank,
#                                    KNeighborsClassifier(n_neighbors=9), "knn_value_noblank.sav")  # .57
# print(acc_value_noblank)

# load model
# model = pickle.load(open(filename,'rb'))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


'''input_shape = [X.shape[1]]  # shit need to be a list
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    # layers.BatchNormalization(),
    # layers.Dropout(.3),
    layers.Dense(256, activation='relu'),
    # layers.BatchNormalization(),
    # layers.Dropout(.3),
    layers.Dense(10, activation='sigmoid')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

X_train, X_valid, y_train, y_valid = train_test_split(X, y_value, test_size=.15, random_state=42)

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

# model with .937 of val_accuracy after 300 epochs
history_df = pd.DataFrame(history.history)
history_df.to_csv("doku_ds/keras_model_history.csv", index=False)

model.save('save_model\digit_recognizer_300')'''
load_model = tf.keras.models.load_model('save_model\digit_recognizer_300')
# load_model.summary()

X_test = df_test.drop(columns=["value", "is_blank"])
y_blank_test = df_test['is_blank']
y_value_test = df_test['value']

# load_model.evaluate(X_test, y_value_test)


import document_scanner
import Solver
board = document_scanner.generate_board_df('v2_test/image83.jpg')
board_raw = load_model(board.values)
board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
print(board_clean)
Solver.run_solver(board_clean)

# to do list :
# * Find how to use keras model to make predictions
# * Maybe try to filer blanks with knn before feeding to keras layers
# * recreate board out of predictions


