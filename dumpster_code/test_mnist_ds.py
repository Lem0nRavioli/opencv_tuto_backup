import idx2numpy
import numpy as np
import cv2
import pandas as pd


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


train_df = generate_df('../mnist_dataset/train-images-idx3-ubyte', '../mnist_dataset/train-labels-idx1-ubyte')
test_df = generate_df('../mnist_dataset/t10k-images-idx3-ubyte', '../mnist_dataset/t10k-labels-idx1-ubyte')
doku_df_train = pd.read_csv('../doku_ds/sudoku_train_28x28_noise.csv')
doku_df_train = doku_df_train.drop(columns=['value', 'is_blank'])
train_df = train_df.drop(columns=['value'])
# map = lambda x: np.random.randint(140, 160) if x > 200 else np.random.randint(90, 110)
# train_df.applymap(map)

#
# print(train_df.head())
# print(train_df.shape)
# print(test_df.head())
# print(test_df.shape)
# print(doku_df_train.head())
print(np.array(train_df.loc[0]).reshape((28, 28)))
print(np.array(doku_df_train.loc[7006]).reshape((28, 28)))
print(np.random.randint(140, 160))
# print(doku_df_train.shape)

# train_filepath = '../mnist_dataset/train-images-idx3-ubyte'
# train_raw = idx2numpy.convert_from_file(train_filepath)
# # put this in format white background black number
# train_raw = np.uint8(abs(np.int32(train_raw) - 255))
# # print(train_raw.shape)
# df_train = pd.DataFrame(train_raw.reshape((60000, 784)))
# print(df_train.head())
# print(df_train.shape)
#
# # print(train_raw[4])
# # cv2.imshow("image", arr[4])
# # cv2.waitKey()