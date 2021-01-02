from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import relu, softmax

model = Sequential([
    Dense(units=64, activation=relu, input_shape=(784,)),
    Dense(units=10, activation=softmax)
])

model2 = Sequential()
model2.add(Dense(units=64, activation=relu, input_shape=(128,)))
model2.add(Dense(units=10, activation=softmax))

model3 = Sequential([
    Flatten(input_shape=(32, 32)),
    Dense(units=64, activation=relu),
    Dense(units=10, activation=softmax)
])

simple_model = Sequential([
    Flatten(),
    Dense(32)
])

print((64 + 1) * 16 + (16 + 1) * 16 * 2 + 17 * 8)
