from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.activations import relu, softmax, sigmoid

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=16, activation=relu, name='layer_1'),
    Dense(units=32, activation=relu, name='layer_2'),
    Dense(units=10, activation=relu),
    Dense(units=15, activation=sigmoid),
    Softmax()
])

# print(model.weights)
# ValueError: Weights for model sequential have not yet been created.
# Weights are created when the Model is first called on inputs or `build()` is called with an
# `input_shape`.

model.summary()
