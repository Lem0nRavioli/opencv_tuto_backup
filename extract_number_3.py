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
import doc_scan_3 as scanner

# pd.options.display.width = None


def generate_df(data_path, labels_path):
    data_raw = idx2numpy.convert_from_file(data_path)
    labels = idx2numpy.convert_from_file(labels_path)
    df = pd.DataFrame(data_raw.reshape((data_raw.shape[0], 784)))
    df['value'] = labels
    df.loc[df['value'] == 0, df.columns[:784]] = 0
    return df


train_df_mnist = generate_df('mnist_dataset/train-images-idx3-ubyte', 'mnist_dataset/train-labels-idx1-ubyte')
test_df_mnist = generate_df('mnist_dataset/t10k-images-idx3-ubyte', 'mnist_dataset/t10k-labels-idx1-ubyte')
test_path = "v2_train/image58.jpg"
path = "test_pic"
image_file = "test_board_1.jpg"
img_path = os.path.join(path, image_file)
final = scanner.extract_sudoku(img_path)

# to do :
# * cut the board into 9x9 30x30 pxl
# * crop them [1:29, 1:29]
# * train a model with mnist
# * feed the 9x9 [1:29, 1:29] board to the model
# * select argmax and reshape 9x9 answer
# * call solver with board
# * popcorn