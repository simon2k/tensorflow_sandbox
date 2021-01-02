from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.activations import relu, elu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy, MeanAbsoluteError
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation=relu, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(3, 3)),
    Flatten(),
    Dense(units=10, activation=softmax)
])

model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.002)
accuracy = SparseCategoricalAccuracy()
mae = MeanAbsoluteError()

model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=[accuracy, mae])
print(model.optimizer)
print(model.optimizer.lr)
print(model.loss)
print(model.metrics)

# Load data
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_imgs, train_lbls), (test_imgs, test_lbls) = fashion_mnist_data.load_data()

print(train_imgs.shape)
print(train_lbls.shape)

labels = [
    'T-shirt',
    'Trousers',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Scale data - unifity data

train_imgs = train_imgs / 255.
test_imgs = test_imgs / 255.

# test_img = test_imgs[30]
# print(test_img.shape)
# print(np.expand_dims(test_img, axis=-1).shape)
# print(np.expand_dims(np.expand_dims(test_img, axis=-1), axis=0).shape)

# print(labels[train_lbls[0]])
# plt.imshow(test_imgs[0])
# plt.show()

# model.fit(np.expand_dims(train_imgs, axis=-1), train_lbls, epochs=2, batch_size=256)
# model.fit(train_imgs[..., np.newaxis], train_lbls, epochs=2, batch_size=256, verbose=2)
history = model.fit(train_imgs[..., np.newaxis], train_lbls, epochs=3, batch_size=32, verbose=1)

df = pd.DataFrame(history.history)
print(df.tail())
print(df.head())

# loss_plot = df.plot(y='loss', title='Loss vs Epochs')
# loss_plot.set(xlabel='Epochs', ylabel='Loss')
#
# accuracy_plot = df.plot(y='sparse_categorical_accuracy', title='Acc vs Epochs')
# accuracy_plot.set(xlabel='Epochs', ylabel='Acc')
#
# accuracy_plot = df.plot(y='mean_absolute_error', title='MAE vs Epochs')
# accuracy_plot.set(xlabel='Epochs', ylabel='MAE')

plt.show()

evaluation = model.evaluate(test_imgs[..., np.newaxis], test_lbls, verbose=2)
print('evaluation: ', evaluation)

idx = 10
test_img = test_imgs[idx]
print('label: ', labels[test_lbls[idx]])
predictions = model.predict(np.expand_dims(np.expand_dims(test_img, axis=-1), axis=0))[0]
prediction = np.argmax(predictions)
print('prediction: ', prediction, labels[prediction])
plt.imshow(test_img)

Conv2D(filters=8, kernel_size=(3, 3), padding='SAME', activation=relu)

MaxPooling2D(pool_size=(2, 2))
