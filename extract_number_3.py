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
import doc_scan_3

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