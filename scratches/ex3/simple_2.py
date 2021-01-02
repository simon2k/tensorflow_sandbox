import tensorflow as tf
from tensorflow.keras.activations import relu, softmax, elu, selu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.initializers import Constant, RandomNormal, RandomUniform, Zeros, HeUniform, Ones, Orthogonal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

model = Sequential([
    Conv1D(
        filters=16,
        kernel_size=3,
        input_shape=(128, 64),
        kernel_initializer=RandomUniform(),
        bias_initializer=Zeros,
        activation=relu),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(units=64,
          kernel_initializer=HeUniform(),
          bias_initializer=Ones(),
          activation=relu)
])

model.summary()

model.add(Dense(units=64,
                kernel_initializer=RandomNormal(mean=0., stddev=0.005),
                bias_initializer=Constant(value=0.03),
                activation=elu,
                name='rand_norm_0.0005'))

model.add(Dense(units=8,
                kernel_initializer=Orthogonal(gain=0.9),
                bias_initializer=Constant(value=0.04),
                activation=selu))

model.summary()


def my_custom_initializer(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)


model.add(Dense(units=64, kernel_initializer=my_custom_initializer, activation=selu))

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy()])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'mae'])

print(model.optimizer)
print(model.loss)
print(model.metrics)
print(model.optimizer.lr)
