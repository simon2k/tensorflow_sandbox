import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.activations import relu, softmax


def load_eurosat_data():
    data_dir = 'data/'
    X_train = np.load(os.path.join(data_dir, 'x_train.npy')) / 255.
    X_test = np.load(os.path.join(data_dir, 'x_test.npy')) / 255.
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return (X_train, y_train), (X_test, y_test)


def get_new_model(input_shape):
    model = Sequential([
        Conv2D(filters=16, input_shape=input_shape, kernel_size=(3, 3), padding='SAME', activation=relu, name='conv_1'),
        Conv2D(filters=8, kernel_size=3, activation=relu, padding='SAME', name='conv_2'),
        MaxPooling2D(pool_size=8, name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation=relu, name='dense_1'),
        Dense(units=10, activation=softmax, name='dense_2')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_test_accuracy(model, data, targets):
    test_loss, test_accuracy = model.evaluate(x=data, y=targets)
    print(f'Loss: {np.round(test_loss, 3)} Accuracy: {np.round(test_accuracy, 3)}')


def get_checkpoint_every_epoch():
    return ModelCheckpoint(filepath='checkpoints_every_epoch/checkpoint_{epoch}',
                           save_freq='epoch',
                           save_weights_only=True,
                           vebose=1)


def get_checkpoint_bets_only():
    return ModelCheckpoint(filepath='checkpoints_best_only/checkpoint',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max',
                           monitor='val_accuracy',
                           vebose=1)


def get_early_stopping():
    return EarlyStopping(patience=3, monitor='val_accuracy', verbose=1)


def get_model_last_epoch(model):
    model.load_weights(tf.train.latest_checkpoint('./checkpoints_every_epoch'))
    return model


def get_model_best_epoch(model):
    model.load_weights(tf.train.latest_checkpoint('./checkpoints_best_only'))
    return model


def get_model_eurosatnet():
    model = load_model('./models/EuroSatNet.h5')
    return model


(X_train, y_train), (X_test, y_test) = load_eurosat_data()
model = get_new_model(X_train[0].shape)
model.summary()

print('Untrained model accuracy:')
get_test_accuracy(model, X_test, y_test)

callbacks = [get_checkpoint_every_epoch(), get_checkpoint_bets_only(), get_early_stopping()]
model.fit(x=X_train, y=y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

print('Trained model accuracy:')
get_test_accuracy(model, X_test, y_test)

model_last_epoch = get_model_last_epoch(get_new_model(X_train[0].shape))
model_best_epoch = get_model_best_epoch(get_new_model(X_train[0].shape))

print('Best epoch accuracy: ')
get_test_accuracy(model_best_epoch, X_test, y_test)

print('Latest epoch accuracy: ')
get_test_accuracy(model_last_epoch, X_test, y_test)

model_eurosatnet = get_model_eurosatnet()
model_eurosatnet.summary()
get_test_accuracy(model_eurosatnet, X_test, y_test)
