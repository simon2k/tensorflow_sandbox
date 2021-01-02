import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import elu, relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, BinaryAccuracy, SparseTopKCategoricalAccuracy
import tensorflow.keras.backend as K

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=32, activation=relu),
    Dense(units=32, activation=elu),
    Dense(units=10, activation=softmax)])

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

# Case 1

y_true = tf.constant([0., 1., 1.])
y_pred = tf.constant([.4, .8, .3])
print('y pred round: ', K.round(y_pred))
accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
print('acc: ', accuracy)

# Case 2

# Binary

y_true = tf.constant([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
y_pred = tf.constant([[0.4, 0.6], [0.3, 0.7], [0.05, 0.95], [0.33, 0.67]])
accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
print('acc2: ', accuracy)

# Categorical classification, m>2
y_true = tf.constant([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
y_pred = tf.constant([[0.4, 0.6, 0.0, 0.0], [0.3, 0.2, 0.1, 0.4], [0.05, 0.35, 0.5, 0.1]])
accuracy = K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
print('acc3: ', accuracy)

# Setting threshold for value to be rounded as 1 in the sparse categorical crossentropy

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[BinaryAccuracy(threshold=0.7)])

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])


def custom_metric(y_true, y_pred):
    K.mean(K.equal(y_true, K.round(y_pred)))


model.comile(optimizer='adam',
             loss=SparseCategoricalCrossentropy(),
             metrics=['accuracy', SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy(k=7), custom_metric])
