import tensorflow as tf
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np

print('list_physical_devices: ', tf.config.list_physical_devices())
print('list_physical_devices CPU: ', tf.config.list_physical_devices('GPU'))
print('list_physical_devices GPU: ', tf.config.list_physical_devices('CPU'))

x = tf.random.uniform(shape=(3, 3))

print('x device: ', x.device)


def time_matadd(x):
    start = time.time_ns()

    for loop in range(10):
        tf.add(x, x)
    result = time.time_ns() - start

    print(f'Mtx addition (10 loops: {result} ns')


def time_matmul(x):
    start = time.time_ns()

    for loop in range(10):
        tf.matmul(x, x)
    result = time.time_ns() - start

    print(f'Mtx matmul (10 loops: {result} ns')


# print('on CPU')
# with tf.device('CPU:0'):
#     x = tf.random.uniform(shape=(1000, 1000))
#     assert x.device.endswith('CPU:0')
#     time_matadd(x)
#     time_matadd(x)
#     time_matadd(x)
#     time_matmul(x)
#     time_matmul(x)
#     time_matmul(x)
#
# print('on GPU')
# with tf.device('GPU:0'):
#     x = tf.random.uniform(shape=(1000, 1000))
#     assert x.device.endswith('GPU:0')
#     time_matadd(x)
#     time_matadd(x)
#     time_matadd(x)
#     time_matmul(x)
#     time_matmul(x)
#     time_matmul(x)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255., X_test / 255.


def build_model(input_shape=(28, 28, 1)):
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=64, activation=relu),
        Dense(units=10, activation=softmax)
    ])
    return model

with tf.device('CPU:0'):
    model = build_model()
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    start = time.time_ns()
    model.fit(X_train[..., np.newaxis], y_train, epochs=10)
    result = time.time_ns() - start
    print(f'CPU training timme: {result}ns')

with tf.device('GPU:0'):
    model = build_model()
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    start = time.time_ns()
    model.fit(X_train[..., np.newaxis], y_train, epochs=10)
    result = time.time_ns() - start
    print(f'GPU training timme: {result}ns')
