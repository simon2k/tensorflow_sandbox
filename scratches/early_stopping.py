from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.activations import elu, relu, softmax
from tensorflow.keras.metrics import SparseCategoricalAccuracy, MeanAbsoluteError
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# validation_split=.2
# validation_data=(X_valid, y_valid)
# train_test_split(X, y, test_size=.3

diabetes_dataset = load_diabetes()
# print(diabetes_dataset['DESCR'])
#
# Diabetes dataset
# ----------------
#
# Ten baseline variables, age, sex, body mass index, average blood
# pressure, and six blood serum measurements were obtained for each of n =
# 442 diabetes patients, as well as the response of interest, a
# quantitative measure of disease progression one year after baseline.
#
# **Data Set Characteristics:**
#
#   :Number of Instances: 442
#
#   :Number of Attributes: First 10 columns are numeric predictive values
#
#   :Target: Column 11 is a quantitative measure of disease progression one year after baseline
#
#   :Attribute Information:
#       - age     age in years
#       - sex
#       - bmi     body mass index
#       - bp      average blood pressure
#       - s1      tc, T-Cells (a type of white blood cells)
#       - s2      ldl, low-density lipoproteins
#       - s3      hdl, high-density lipoproteins
#       - s4      tch, thyroid stimulating hormone
#       - s5      ltg, lamotrigine
#       - s6      glu, blood sugar level
#
# Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times
# `n_samples` (i.e. the sum of squares of each column totals 1).
#
# Source URL:
# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
#
# For more information see:
# Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression,"
# Annals of Statistics (with discussion), 407-499.
# (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)

# print(diabetes_dataset.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
# print(targets)

# normalize targets
targets = (targets - targets.mean(axis=0)) / targets.std()
# print(targets)

train_dataset, test_dataset, train_targets, test_targets = train_test_split(data, targets, test_size=.1)


# print(train_dataset.shape)
# print(test_dataset.shape)
# print(train_targets.shape)
# print(test_targets.shape)

# model

# L2 = weight decay

def get_model(train_dataset):
    model = Sequential([
        Dense(units=32, activation=relu, input_shape=(train_dataset.shape[1],)),
        Dropout(rate=.5),
        Dense(units=32,
              activation=relu,
              kernel_regularizer=l2(l2=.001),
              bias_regularizer=l1(l1=.002)),
        Dropout(rate=.5),
        Dense(units=32, activation=relu, kernel_regularizer=l1(l1=.001)),
        Dropout(rate=.5),
        Dense(units=32, activation=relu, kernel_regularizer=l1_l2(l1=.001, l2=.002)),
        Dense(units=32, activation=relu, kernel_regularizer=l1(l1=.001)),
        Dense(units=32, activation=relu, kernel_regularizer=l2(l2=.005)),
        Dense(units=1)
    ])
    model.summary()
    return model


# wd = weight decay
def get_regularized_model(train_dataset, wd, rate):
    model = Sequential([
        Dense(units=128, activation=relu, input_shape=(train_dataset.shape[1],)),
        Dropout(rate=rate),
        Dense(units=128, activation=relu, kernel_regularizer=l2(l2=wd)),
        Dropout(rate=rate),
        Dense(units=128, activation=relu, kernel_regularizer=l2(l2=wd)),
        Dropout(rate=rate),
        Dense(units=128, activation=relu, kernel_regularizer=l2(l2=wd)),
        Dropout(rate=rate),
        Dense(units=128, activation=relu, kernel_regularizer=l2(l2=wd)),
        Dropout(rate=rate),
        Dense(units=128, activation=relu, kernel_regularizer=l2(l2=wd)),
        Dropout(rate=rate),
        Dense(units=1)
    ])
    return model


def compile_model(model):
    model.compile(
        optimizer=Adam(),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )


class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print('Starting training...')

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Starting the epoch {epoch}')

    def on_train_batch_begin(self, batch, logs=None):
        print(f'Starting the batch {batch}')

    def on_train_batch_end(self, batch, logs=None):
        print(f'Ending training batch {batch}')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Ending epoch {epoch}')

    def on_train_end(self, logs=None):
        print('Finished training...', logs)


class TestingCallback(Callback):
    def on_train_begin(self, logs=None):
        print('Starting testing...')

    def on_test_batch_begin(self, batch, logs=None):
        print(f'Starting testing a batch {batch}')

    def on_test_batch_end(self, batch, logs=None):
        print(f'Ending testing a batch {batch}')

    def on_test_end(self, logs=None):
        print('Finished testing...', logs)


class PredictionCallback(Callback):
    def on_predict_begin(self, logs=None):
        print('Starting prediction...')

    def on_predict_batch_begin(self, batch, logs=None):
        print(f'Starting prediction on a batch {batch}')

    def on_predict_batch_end(self, batch, logs=None):
        print(f'Ending prediction on a batch {batch}')

    def on_predict_end(self, logs=None):
        print('Finished prediction...', logs)


training_callback = TrainingCallback()
testing_callback = TestingCallback()
prediction_callback = PredictionCallback()

unregularized_model = get_model(train_dataset)
regularized_model = get_regularized_model(train_dataset, wd=1e-5, rate=.3)

compile_model(unregularized_model)
compile_model(regularized_model)


def train_model(model):
    history = model.fit(train_dataset,
                        train_targets,
                        epochs=100,
                        validation_split=.15,
                        batch_size=64,
                        verbose=False,
                        callbacks=[
                            training_callback,
                            EarlyStopping(patience=10, monitor='val_mean_absolute_error', min_delta=.01, mode='min')
                        ])
    return history

unregularized_history = train_model(unregularized_model)
regularized_history = train_model(regularized_model)

print('Unregularized')
unregularized_model.evaluate(test_dataset, test_targets, callbacks=[testing_callback])  # no dropout - also in model.predict
print('Regularized')
regularized_model.evaluate(test_dataset, test_targets, callbacks=[testing_callback])  # no dropout - also in model.predict
# Unregularized 28 epochs Finished testing... {'loss': 0.9153466820716858, 'mean_absolute_error': 0.6878498196601868}
# Regularized 7 epochs Finished testing... {'loss': 0.4595632255077362, 'mean_absolute_error': 0.5690906643867493}

fig = plt.figure(figsize=(12, 10))
fig.add_subplot(221)
plt.plot(unregularized_history.history['loss'])
plt.plot(unregularized_history.history['val_loss'])
plt.title('Unregularized Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(222)
plt.plot(regularized_history.history['loss'])
plt.plot(regularized_history.history['val_loss'])
plt.title('Regularized Loss vs Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(223)
plt.plot(unregularized_history.history['mean_absolute_error'])
plt.plot(unregularized_history.history['val_mean_absolute_error'])
plt.title('Unregularized MAE vs Epochs')
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(224)
plt.plot(regularized_history.history['mean_absolute_error'])
plt.plot(regularized_history.history['val_mean_absolute_error'])
plt.title('Regularized MAE vs Epochs')
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

idx = 0
values = test_dataset[idx]
# print(values.shape, values[..., np.newaxis].shape)

unreg_prediction = unregularized_model.predict(values[np.newaxis, ...], callbacks=[prediction_callback])[0]
reg_prediction = regularized_model.predict(values[np.newaxis, ...], callbacks=[prediction_callback])[0]
print(unreg_prediction, reg_prediction, test_targets[0])
