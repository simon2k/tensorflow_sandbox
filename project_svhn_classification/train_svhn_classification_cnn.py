import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Load data
train = loadmat('./assets/train_32x32.mat')
test = loadmat('./assets/test_32x32.mat')

X_train = train['X']
y_train = train['y']
X_test = test['X']
y_test = test['y']

print('Unique labels for train data: ', np.unique(y_train))
print('Unique labels for test data: ', np.unique(y_test))

print('X train shape: ', X_train.shape)
print('X test shape: ', X_test.shape)

# Replace the label "10" with "0" and categorize data
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

print('Unique labels for train data after replacement: ', np.unique(y_train))
print('Unique labels for test data after replacement: ', np.unique(y_test))

# Move the samples axis as first
X_train = np.moveaxis(X_train, -1, 0)
X_test = np.moveaxis(X_test, -1, 0)

# Display sample 10 images from the "train" set
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 5))

for i in range(10):
    idx = np.random.choice(range(X_train.shape[0]))
    ax[i].set_axis_off()
    ax[i].imshow(X_train[idx])
    ax[i].set_title(y_train[idx])

plt.show()

X_train_grayscale = np.mean(X_train, axis=3, keepdims=True) / 255.
X_test_grayscale = np.mean(X_test, axis=3, keepdims=True) / 255.

print('X_train_grayscale shape: ', X_train_grayscale.shape)
print('X_test_grayscale shape: ', X_test_grayscale.shape)

# Display sample 10 images from the "train" set after normalization
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 5))

for i in range(10):
    idx = np.random.choice(range(X_train_grayscale.shape[0]))
    ax[i].set_axis_off()
    ax[i].imshow(X_train_grayscale[idx, :, :, 0], cmap='gray')
    ax[i].set_title(y_train[idx])

plt.show()


def get_cnn_model():
    model = Sequential([
        Conv2D(filters=16,
               kernel_size=(3, 3),
               input_shape=X_train_grayscale[0].shape,
               padding='same',
               activation=relu,
               kernel_initializer=he_uniform(),
               kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3),
               bias_initializer=he_uniform(),
               bias_regularizer=l1_l2(l1=1e-3, l2=1e-3),
               name='conv2d_1'),
        Conv2D(filters=8,
               kernel_size=(3, 3),
               padding='same',
               activation=relu,
               kernel_initializer=he_uniform(),
               kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3),
               bias_initializer=he_uniform(),
               bias_regularizer=l1_l2(l1=1e-3, l2=1e-3),
               name='conv2d_2'),
        MaxPooling2D(pool_size=(3, 3), name='max_pooling_2d_1'),
        BatchNormalization(name='batch_normalization_1'),
        Dropout(rate=0.3, name='dropout_1'),
        Conv2D(filters=8,
               kernel_size=(3, 3),
               padding='same',
               activation=relu,
               kernel_initializer=he_uniform(),
               kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3),
               bias_initializer=he_uniform(),
               bias_regularizer=l1_l2(l1=1e-3, l2=1e-3),
               name='conv2d_3'),
        MaxPooling2D(pool_size=(3, 3), name='max_pooling_2d_2'),
        Flatten(name='flatten'),
        Dense(units=32,
              activation=relu,
              kernel_initializer=he_uniform(),
              kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3),
              bias_initializer=he_uniform(),
              bias_regularizer=l1_l2(l1=1e-3, l2=1e-3),
              name='dense_1'),
        Dropout(0.3, name='dropout_2'),
        Dense(units=16,
              activation=relu,
              kernel_initializer=he_uniform(),
              bias_initializer=he_uniform(),
              kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3),
              name='dense_2'),
        Dense(units=10,
              activation=softmax,
              kernel_initializer=he_uniform(),
              kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3),
              bias_initializer=he_uniform(),
              bias_regularizer=l1_l2(l1=1e-3, l2=1e-3),
              name='dense_output')
    ])
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    return model


def evaluate_model(model, model_name):
    test_loss, test_accuracy = model.evaluate(x=X_test_grayscale, y=y_test)
    print(f'Evaluating model {model_name}'
          f'\n* Val loss: {np.round(test_loss, 3)}'
          f'\n* Val accuracy: {np.round(test_accuracy, 3)}')


def display_model_performance(history):
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='lower right')

    fig.add_subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


cnn_model = get_cnn_model()
cnn_model.summary()

checkpoint_callback = ModelCheckpoint(filepath='best_accurate_cnn_model/model_{epoch}',
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='max',
                                      verbose=1)

early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=5, mode='min', verbose=1)

reduce_lr_on_plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=1e-3, patience=5, mode='min', verbose=1)

history = cnn_model.fit(x=X_train_grayscale,
                        y=y_train,
                        epochs=30,
                        validation_data=(X_test_grayscale, y_test),
                        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_on_plateau_callback])

evaluate_model(cnn_model, model_name='CNN')
# Evaluating model CNN
# * Val loss: 0.932
# * Val accuracy: 0.811

display_model_performance(history)
