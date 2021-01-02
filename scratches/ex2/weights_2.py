import tensorflow as tf
from tensorflow.keras.activations import relu, softmax, elu, selu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.initializers import Constant, RandomNormal, RandomUniform, Zeros, HeUniform, Ones, Orthogonal
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

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

weight_layers = [layer for layer in model.layers if len(layer.weights) > 0]

for i, layer in enumerate(weight_layers):
    for j in [0, 1]:
        axes[i, j].hist(layer.weights[j].numpy().flatten(), align='left')
        axes[i, j].set_title(layer.weights[j].name)

plt.show()
