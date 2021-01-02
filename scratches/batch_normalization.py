import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

diabetes_dataset = load_diabetes()
diabetes_data = diabetes_dataset['data']
diabetes_target = diabetes_dataset['target']

diabetes_target = (diabetes_target - diabetes_target.mean(axis=0)) / diabetes_target.std()

X_train, X_test, y_train, y_test = train_test_split(diabetes_data, diabetes_target, test_size=.1)

model = Sequential([
    Dense(units=64, input_shape=(X_train.shape[1],), activation=relu),
    BatchNormalization(),
    Dropout(.5),
    BatchNormalization(),
    Dropout(.5),
    Dense(units=128, activation=relu)
])

model.summary()

model.add(BatchNormalization(
    momentum=.95,  # momentum for moving average
    epsilon=.005,  # small float added to variance to avoid div. by zero
    axis=-1,  # axis that should be normalized
    beta_initializer=RandomNormal(mean=0., stddev=.05),  # init for beta weight
    gamma_initializer=Constant(value=.9)  # init for gamma weight
))

model.add(Dense(units=1))

model.summary()

model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

history = model.fit(X_train, y_train, epochs=100, validation_split=.15, batch_size=64)

frame = pd.DataFrame(history.history)
epochs = np.arange(len(frame))

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121)
ax.plot(epochs, frame['loss'], label='Train')
ax.plot(epochs, frame['val_loss'], label='Validation')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss over Epochs')
ax.legend()

ax = fig.add_subplot(122)
ax.plot(epochs, frame['mean_absolute_error'], label='Train')
ax.plot(epochs, frame['val_mean_absolute_error'], label='Validation')
ax.set_xlabel('Epochs')
ax.set_ylabel('MAE')
ax.set_title('MAE over Epochs')
ax.legend()
plt.show()
