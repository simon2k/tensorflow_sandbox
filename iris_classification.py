import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_uniform, ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def read_in_and_split_data(iris_data):
    train_data, test_data, train_targets, test_targets = train_test_split(
        iris_data['data'],
        iris_data['target'],
        test_size=.1)
    return train_data, test_data, train_targets, test_targets


def get_model(input_shape):
    model = Sequential([
        Dense(units=64,
              input_shape=input_shape,
              kernel_initializer=he_uniform(),
              bias_initializer=ones(),
              activation=relu),
        Dense(units=128, activation=relu),
        Dense(units=128, activation=relu),
        Dense(units=128, activation=relu),
        Dense(units=128, activation=relu),
        Dense(units=64, activation=relu),
        Dense(units=64, activation=relu),
        Dense(units=64, activation=relu),
        Dense(units=64, activation=relu),
        Dense(units=3, activation=softmax)
    ])
    return model


def get_regularized_model(input_shape, dropout_rate, weight_decay):
    model = Sequential([
        Dense(units=64,
              input_shape=input_shape,
              kernel_initializer=he_uniform(),
              bias_initializer=ones(),
              activation=relu,
              kernel_regularizer=l2(weight_decay)),
        Dense(units=128, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dense(units=128, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dropout(rate=dropout_rate),
        Dense(units=128, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dense(units=128, activation=relu, kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Dense(units=64, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dense(units=64, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dropout(rate=dropout_rate),
        Dense(units=64, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dense(units=64, activation=relu, kernel_regularizer=l2(weight_decay)),
        Dense(units=3, activation=softmax)
    ])
    return model


def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )


def train_model(model, train_data, train_targets, epochs):
    history = model.fit(train_data, train_targets, epochs=epochs, validation_data=(test_data, test_targets))
    return history


iris_data = load_iris()
(train_data, test_data, train_targets, test_targets) = read_in_and_split_data(iris_data)

train_targets = to_categorical(train_targets)
test_targets = to_categorical(test_targets)

model = get_model(train_data[0].shape)

compile_model(model)

history = train_model(model, train_data, train_targets, epochs=800)

fig = plt.figure(figsize=(12, 5))

fig.add_subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

fig.add_subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.show()
