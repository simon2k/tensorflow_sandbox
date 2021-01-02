from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import RandomNormal, Zeros, RandomUniform, Ones, Initializer
import tensorflow as tf


def custom_initializer(shape, dtype=None):
    return tf.random.normal(shape, dtype=dtype)


class BestRandomNormal(Initializer):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    def get_config(self):
        return {'mean': self.mean, 'stddev': self.stddev}


model = Sequential([
    Conv2D(
        filters=16,
        kernel_size=(3, 3),
        input_shape=(192, 192, 3),
        activation=relu,
        bias_initializer=Zeros(),
        kernel_initializer=RandomNormal(stddev=1., mean=0., seed=None),
        padding='same'),
    MaxPooling2D(pool_size=(3, 3), padding='same'),
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation=relu,
        bias_initializer=Zeros(),
        kernel_initializer=RandomUniform(minval=-0.06, maxval=0.06, seed=None),
        padding='same'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation=relu,
        bias_initializer=Ones(),
        kernel_initializer=custom_initializer,
        padding='same'),
    MaxPooling2D(pool_size=(3, 3), padding='SAME'),
])

model.summary()
