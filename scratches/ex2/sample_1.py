from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.activations import relu, softmax

model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation=relu, input_shape=(32, 32, 3), padding='SAME'),
    MaxPooling2D(pool_size=(3, 3)),
    Flatten(),
    Dense(units=64, activation=relu),
    Dense(units=10, activation=softmax)
])

model.summary()

model_2 = Sequential([
    Conv2D(filters=16, kernel_size=3, activation=relu, input_shape=(32, 32, 3), padding='SAME'),
    MaxPooling2D(pool_size=3),
    Flatten(),
    Dense(units=64, activation=relu),
    Dense(units=10, activation=softmax)
])

model_2.summary()
