import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_uniform, ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, CategoricalCrossentropy as CategoricalCrossentropyAcc
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.
X_test = X_test / 255.

fix, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 5))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(X_train[i])


# plt.show()


def get_test_accuracy(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)
    print(f'Loss: {np.round(test_loss, 3)} accuracy: {np.round(test_accuracy, 3)}')


def get_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3), activation=relu, name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation=relu, name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation=relu, name='dense_1'),
        Dense(units=10, activation=softmax, name='dense_2')
    ])
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model


model = get_model()

model.summary()

get_test_accuracy(model, X_test, y_test)  # Loss: 2.359 accuracy: 0.112

# Checkpoint for storing weights

checkpoint_path = './model_checkpoints/checkpoint'
# adding .h5 saves as a single file ending with h5 (HDF file - Hierarchical Data Format file)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', save_weights_only=True, verbose=1)

# save_freq - if integer, then it's a number of samples - in this case after seeing 1000 since the last model save
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_freq=1000, save_weights_only=True, verbose=1)

# To save the best weights, based on the monitor - by default `val_loss`
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

# `mode` - either maximize or minimize
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1,
                                      mode='max', monitor='val_accuracy')

# Filename to include epoch & batch number:
checkpoint_callback = ModelCheckpoint(filepath='training_assets/model.{epoch}.{batch}',
                                      save_best_only=True, save_weights_only=True, verbose=1,
                                      mode='max', monitor='val_accuracy', )

# Filename to include epoch & batch number:
checkpoint_callback = ModelCheckpoint(filepath='training_assets/model.{epoch}-{val_loss}',
                                      save_best_only=True, save_weights_only=True, verbose=1,
                                      mode='max', monitor='val_accuracy')

checkpoint_callback = ModelCheckpoint(
    filepath='training_assets/model-{epoch}',
    save_weights_only=True,
    verbose=1,
    mode='max',
    save_freq=1000,
    monitor='val_accuracy')

checkpoint_best_callback = ModelCheckpoint(
    filepath='training_assets_best/model-{epoch}',
    save_freq='epoch',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

checkpoint_whole_model_callback = ModelCheckpoint(
    filepath='training_assets_entire_model/model-{epoch}',
    save_weights_only=False,  # default
    verbose=1,
    mode='max',
    save_freq='epoch',
    monitor='val_accuracy')

# model.fit(x=X_train, y=y_train, epochs=3, batch_size=64, callbacks=[checkpoint_callback])

# get_test_accuracy(model, X_test, y_test)  # Loss: 1.236 accuracy: 0.567

model = get_model()

# get_test_accuracy(model, X_test, y_test)  # Loss: 2.359 accuracy: 0.101
#
# model.load_weights('./model_checkpoints/checkpoint')
#
# get_test_accuracy(model, X_test, y_test)  # Loss: 1.236 accuracy: 0.567

# model.fit(x=X_train, y=y_train, epochs=3, batch_size=4, callbacks=[checkpoint_callback],
#           validation_data=(X_test, y_test))

X_train = X_train[:100]
y_train = y_train[:100]

history = model.fit(x=X_train, y=y_train,
                    epochs=2,
                    batch_size=16,
                    callbacks=[checkpoint_whole_model_callback],
                    validation_data=(X_test, y_test))

get_test_accuracy(model, X_test, y_test)

df = pd.DataFrame(history.history)
df.plot(y=['accuracy', 'val_accuracy'])

# model.save('entire_model') # to save entire model
model.save('entire_model.h5')  # to save entire model

model = load_model('./entire_model.h5')

get_test_accuracy(model, X_test, y_test)
