from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.activations import relu, sigmoid, linear
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, MeanAbsoluteError

model = Sequential([
    Dense(input_shape=(32,), activation=relu, units=64),
    Dense(units=1, activation=linear)
])

model.summary()

model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),  # adam, rsmprop, adadelta
    loss=BinaryCrossentropy(from_logits=True),
    # mean_squared_error, categorical_crossentropy, sparse_cateogrical_crossentropy
    metrics=[BinaryAccuracy(threshold=0.7), MeanAbsoluteError()])
# metrics=['accuracy']) # mae = mean absolute error - tracks these metrics as an addition to loss

model.summary()
