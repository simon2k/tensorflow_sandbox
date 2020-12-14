import numpy as np
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1]),
    keras.layers.Dense(units=10, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

x = np.array([1,  2,   3,  4,   5],  dtype=float)
y = np.array([1., 1.5, 2., 2.5, 3.], dtype=float)

model.fit(x, y, epochs=500)

print(model.predict([6.]))
