from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax

model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation=relu, input_shape=(3, 28, 28), padding='SAME', strides=2,
           data_format='channels_first'),
    MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_first'),
    Flatten(),
    Dense(units=16, activation=relu),
    Dense(units=10, activation=softmax)
])

model.summary()
