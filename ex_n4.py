import cv2
import numpy as np
import tensorflow as tf
import doc_scan_3 as scanner

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)  # scale data to 1
# x_test = tf.keras.utils.normalize(x_test, axis=1)

for i in range(len(y_train)):
    if y_train[i] == 0:
        x_train[i] = np.zeros((28, 28)).astype(np.float64)
for i in range(len(y_test)):
    if y_test[i] == 0:
        x_test[i] = np.zeros((28, 28)).astype(np.float64)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train_norm = x_train.astype('float32')
x_test_norm = x_test.astype('float32')
x_train_norm = x_train_norm / 255
x_test_norm = x_test_norm / 255


def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # compile model
    opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_better_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
#
# #
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model = define_model()
# model.fit(x_train_norm, y_train, epochs=10, batch_size=32, validation_data=(x_test_norm, y_test))
#
# loss, accuracy = model.evaluate(x_test_norm, y_test)
# print(f"accuracy: {accuracy}, loss:{loss}")

# model.save('save_model\digit_recognizer_basic_1')

# model = define_better_model()
# model.fit(x_train_norm, y_train, epochs=10, batch_size=200, validation_data=(x_test_norm, y_test))
# model.save('save_model\digreco_01')
load_model = tf.keras.models.load_model('save_model\digreco_01')
loss, accuracy = load_model.evaluate(x_test_norm, y_test)
print(f"accuracy: {accuracy}, loss: {loss}")


def generate_board_tiles(path):
    img = scanner.extract_sudoku(path)
    s = 30
    tiles = [img[x + 1:x + s - 1, y + 1:y + s - 1] for x in range(0, img.shape[0], s) for y in
             range(0, img.shape[1], s)]
    return tiles


def generate_board_tiles_2(path):
    img = scanner.extract_sudoku(path)
    img = cv2.resize(img, (900, 900))
    s = 100
    tiles = [img[x:x + s, y:y + s] for x in range(0, img.shape[0], s) for y in
             range(0, img.shape[1], s)]
    tiles_clean = []
    for tile in tiles:
        tile = cv2.resize(tile, (28, 28))
        tiles_clean.append(tile)
    print(np.array(tiles_clean).shape)
    return tiles_clean


img_path = "test_pic/sudoku_shit_angle.jpg"
# img_path = "v2_train/image1.jpg"
tiles = np.array(generate_board_tiles(img_path)).astype(np.float32).reshape((81, 28, 28, 1)) / 255
test = np.array(generate_board_tiles_2(img_path)).astype(np.float32).reshape((81, 28, 28, 1)) / 255

board_raw = load_model(tiles)
board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
print(board_clean)
board_raw = load_model(test)
board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
print(board_clean)

print(x_test_norm.shape)
print(x_test_norm.dtype)
print(tiles.shape)
print(tiles.dtype)
print(test.shape)
print(test.dtype)