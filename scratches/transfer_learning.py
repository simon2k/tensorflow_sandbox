import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf

from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import smart_resize

images_train = np.load('./assets/cats-vs-dogs-dataset/images_train.npy')
images_test = np.load('./assets/cats-vs-dogs-dataset/images_test.npy')
images_valid = np.load('./assets/cats-vs-dogs-dataset/images_valid.npy')
labels_train = np.load('./assets/cats-vs-dogs-dataset/labels_train.npy')
labels_test = np.load('./assets/cats-vs-dogs-dataset/labels_test.npy')
labels_valid = np.load('./assets/cats-vs-dogs-dataset/labels_valid.npy')

print('Train set shape: ', images_train.shape)
print('Train labels shape: ', labels_train.shape)
print('All unique train labels: ', np.unique(labels_train))
print('All unique test labels: ', np.unique(labels_test))
print('All unique valid labels: ', np.unique(labels_valid))


# class_names = np.array(['Dog', 'Cat'])
# plt.figure(figsize=(25, 10))
#
# indices = np.random.choice(images_train.shape[0], size=25, replace=False)
#
# for n, i in enumerate(indices):
#     ax = plt.subplot(5, 5, n+1)
#     plt.imshow(images_train[i])
#     plt.title(class_names[labels_train[i]])
#     plt.axis('off')


def get_model(input_shape):
    input = Input(shape=input_shape)
    h = Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding='SAME')(input)
    h = Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding='SAME')(h)
    h = MaxPooling2D(pool_size=2)(h)
    h = Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding='SAME')(h)
    h = Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding='SAME')(h)
    h = MaxPooling2D(pool_size=2)(h)
    h = Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding='SAME')(h)
    h = Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding='SAME')(h)
    h = MaxPooling2D(pool_size=2)(h)
    h = Flatten()(h)
    h = Dense(units=128, activation=relu)(h)
    h = Dense(units=1, activation=sigmoid)(h)

    model = Model(inputs=input, outputs=h)

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()]
    )

    return model


early_stopping_cb = EarlyStopping(patience=2, verbose=1)

benchmark_model = get_model(images_train[0].shape)
benchmark_model.summary()

model_checkpoint_cb = ModelCheckpoint(
    filepath='./assets/cats_vs_dogs_model/best_model_weights',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

history_benchmark = benchmark_model.fit(
    x=images_train,
    y=labels_train,
    epochs=10,
    batch_size=32,
    validation_data=(images_valid, labels_valid),
    # callbacks=[early_stopping_cb, model_checkpoint_cb]
    callbacks=[model_checkpoint_cb]
)

# print(history_benchmark.history)

plt.figure(figsize=(15, 5))
plt.suptitle('Custom classifier')
plt.subplot(121)
plt.plot(history_benchmark.history['binary_accuracy'])
plt.plot(history_benchmark.history['val_binary_accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_benchmark.history['loss'])
plt.plot(history_benchmark.history['val_loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

benchmark_test_loss, benchmark_test_accuracy = benchmark_model.evaluate(
    x=images_test,
    y=labels_test
)

print(f'Test loss: {np.round(benchmark_test_loss, 2)}')
print(f'Test accuracy: {np.round(benchmark_test_accuracy, 2)}')

mobile_net_v2_model = MobileNetV2()

mobile_net_v2_model.summary()


def build_feature_extractor_model(model):
    input_layer = model.inputs
    output_layer = model.get_layer('global_average_pooling2d').output
    return Model(inputs=input_layer, outputs=output_layer)


feature_extractor = build_feature_extractor_model(mobile_net_v2_model)
print('\nFeature extractor model:\n')
feature_extractor.summary()


def add_new_classifier_head(feature_extractor_model):
    model = Sequential([
        feature_extractor_model,
        Dense(units=32, activation=relu),
        Dropout(rate=.5),
        Dense(units=1, activation=sigmoid)
    ])
    return model


pet_classifier_model = add_new_classifier_head(feature_extractor)
pet_classifier_model.summary()


def freeze_pretrained_weights(model):
    # model.get_layer('model').trainable = False # - the name is assigned a suffix "_n"
    model.layers[0].trainable = False


freeze_pretrained_weights(pet_classifier_model)
pet_classifier_model.summary()

pet_classifier_model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=[BinaryAccuracy()]
)


def resize_images(images):
    return np.array(list(map(
        lambda img: resize(img, output_shape=(224, 224, 3), anti_aliasing=True, preserve_range=True).astype(
            'int32') / 255.,
        images)))


images_valid_mnv2 = resize_images(images_valid)
# print('images_valid_mnv2.shape: ', images_valid_mnv2.shape)
# print('images_valid.shape: ', images_valid.shape)
images_train_mnv2 = resize_images(images_train)
images_test_mnv2 = resize_images(images_test)

# class_names = np.array(['Dog', 'Cat'])
# plt.figure(figsize=(25, 10))
# plt.title('Resized images to 224x224')
#
# indices = np.random.choice(images_valid_mnv2.shape[0], size=25, replace=False)
#
# for n, i in enumerate(indices):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(images_valid_mnv2[i])
#     plt.title(class_names[labels_train[i]])
#     # plt.axis('off')

pet_classifier_history = pet_classifier_model.fit(
    x=images_train_mnv2,
    y=labels_train,
    epochs=10,
    batch_size=64,
    validation_data=(images_valid_mnv2, labels_valid),
    # callbacks=[early_stopping_cb]
)
print('pet_classifier_history:\n', pet_classifier_history.history)
print('history_benchmark:\n', history_benchmark.history)

plt.figure(figsize=(15, 5))
plt.suptitle('Pet classifier model')
plt.subplot(121)
plt.plot(pet_classifier_history.history['binary_accuracy'])
plt.plot(pet_classifier_history.history['val_binary_accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(pet_classifier_history.history['loss'])
plt.plot(pet_classifier_history.history['val_loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

pet_benchmark_test_loss, pet_benchmark_test_accuracy = pet_classifier_model.evaluate(
    x=images_test_mnv2,
    y=labels_test
)

print(f'[PetClass] Test loss: {np.round(benchmark_test_loss, 2)}')
print(f'[PetClass] Test accuracy: {np.round(benchmark_test_accuracy, 2)}')

# Compare both models

benchmark_train_accuracy = history_benchmark.history['binary_accuracy'][-1]
benchmark_val_accuracy = history_benchmark.history['val_binary_accuracy'][-1]
benchmark_train_loss = history_benchmark.history['loss'][-1]
benchmark_val_loss = history_benchmark.history['val_loss'][-1]

pet_benchmark_train_accuracy = pet_classifier_history.history['binary_accuracy'][-1]
pet_benchmark_val_accuracy = pet_classifier_history.history['val_binary_accuracy'][-1]
pet_benchmark_train_loss = pet_classifier_history.history['loss'][-1]
pet_benchmark_val_loss = pet_classifier_history.history['val_loss'][-1]

comparison = pd.DataFrame([
    ['Training loss', benchmark_train_loss, pet_benchmark_train_loss],
    ['Training accuracy', benchmark_train_accuracy, pet_benchmark_train_accuracy],
    ['Validation loss', benchmark_val_loss, pet_benchmark_val_loss],
    ['Validation accuracy', benchmark_val_accuracy, pet_benchmark_val_accuracy],
    ['Test loss', benchmark_test_loss, pet_benchmark_test_loss],
    ['Test accuracy', benchmark_test_loss, pet_benchmark_test_loss],
])

comparison.index = [''] * 6
print(comparison)

plt.figure(figsize=(15, 5))
plt.suptitle('Confusion matrix comparison')

preds = benchmark_model.predict(images_test)
preds = (preds > .5).astype('int32')
cm = confusion_matrix(labels_test, preds)
df_cm = pd.DataFrame(cm, index=['Dog', 'Cat'], columns=['Dog', 'Cat'])
plt.subplot(121)
plt.title('Confusion matrix for benchmark model\n')
sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
plt.ylabel('Prediction')
plt.xlabel('Ground truth')

preds = pet_classifier_model.predict(images_test_mnv2)
preds = (preds > .5).astype('int32')
cm = confusion_matrix(labels_test, preds)
df_cm = pd.DataFrame(cm, index=['Dog', 'Cat'], columns=['Dog', 'Cat'])
plt.subplot(122)
plt.title('Confusion matrix for pet classifier model (transfer learning)\n')
sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
plt.ylabel('Prediction')
plt.xlabel('Ground truth')

plt.show()
