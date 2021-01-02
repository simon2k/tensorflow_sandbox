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
from tensorflow.keras.metrics import MeanAbsoluteError as MeanAbsoluteErrorMetric
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LambdaCallback, LearningRateScheduler, \
    ReduceLROnPlateau, ProgbarLogger

diabetes_dataset = load_diabetes()
diabetes_data = diabetes_dataset['data']
diabetes_target = diabetes_dataset['target']

diabetes_target = (diabetes_target - diabetes_target.mean(axis=0)) / diabetes_target.std()

X_train, X_test, y_train, y_test = train_test_split(diabetes_data, diabetes_target, test_size=.1)

model = Sequential([
    Dense(units=128, activation=relu, input_shape=(X_train.shape[1],)),
    Dense(units=64, activation=relu),
    Dense(units=64, activation=relu),
    Dense(units=64, activation=relu),
    Dense(units=1),
])

model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanAbsoluteErrorMetric(), MeanSquaredErrorMetric()])


# Learning rate scheduler

def lr_schedule(epoch, lr):
    if epoch % 2 == 0:
        return lr
    else:
        return lr + epoch / 1000


# history = model.fit(X_train,
#                     y_train,
#                     epochs=100,
#                     validation_split=.15,
#                     batch_size=64,
#                     verbose=2,
#                     callbacks=[
#                         LearningRateScheduler(lr_schedule, verbose=1),
#                         EarlyStopping(patience=10, monitor='val_mean_absolute_error', min_delta=.01, mode='min')
#                     ])
#
# model.evaluate(X_test, y_test)

# lambda scheduler

# history = model.fit(X_train,
#                     y_train,
#                     epochs=100,
#                     validation_split=.15,
#                     batch_size=64,
#                     verbose=2,
#                     callbacks=[
#                         LearningRateScheduler(lambda lr:  1 / (3 + 5 * lr), verbose=1),
#                         EarlyStopping(patience=10, monitor='val_mean_absolute_error', min_delta=.01, mode='min'),
#                     ])
#
# model.evaluate(X_test, y_test)

# CSV logger

# history = model.fit(X_train,
#                     y_train,
#                     epochs=100,
#                     validation_split=.15,
#                     batch_size=64,
#                     verbose=2,
#                     callbacks=[
#                         LearningRateScheduler(lambda lr:  1 / (3 + 5 * lr), verbose=1),
#                         EarlyStopping(patience=10, monitor='val_mean_absolute_error', min_delta=.01, mode='min'),
#                         CSVLogger(filename='additional_callbacks.csv', separator=',', append=True)
#                     ])
#
# model.evaluate(X_test, y_test)

# Lambda callback

# history = model.fit(X_train,
#                     y_train,
#                     epochs=100,
#                     validation_split=.15,
#                     batch_size=64,
#                     verbose=2,
#                     callbacks=[
#                         LearningRateScheduler(lambda lr: 1 / (3 + 5 * lr), verbose=1),
#                         EarlyStopping(patience=10, monitor='val_mean_absolute_error', min_delta=.01, mode='min'),
#                         CSVLogger(filename='additional_callbacks.csv', separator=',', append=True),
#                         LambdaCallback(
#                             on_epoch_begin=lambda epoch, logs: print(f'Starting epoch {epoch + 1}'),
#                             on_epoch_end=lambda epoch, logs: print(
#                                 f'Finished epoch {epoch + 1} val loss: {logs["val_loss"]}'))
#                     ])

# Reduce LR on plateau

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    validation_split=.15,
                    batch_size=64,
                    verbose=2,
                    callbacks=[
                        EarlyStopping(patience=10, monitor='val_mean_absolute_error', min_delta=.01, mode='min'),
                        CSVLogger(filename='additional_callbacks.csv', separator=',', append=True),
                        ReduceLROnPlateau(factor=.15, patience=6, min_delta=0.01, min_lr=1e-6, verbose=1),
                        ProgbarLogger(count_mode='samples')
                    ])
