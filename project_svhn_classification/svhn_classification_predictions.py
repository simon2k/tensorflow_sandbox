import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(12, 5))

np.set_printoptions(precision=2)


def get_mlp_model():
    model = Sequential([
        Flatten(input_shape=X_train_grayscale[0].shape),
        Dense(units=128, activation=relu),
        Dense(units=64, activation=relu),
        Dense(units=32, activation=relu),
        Dense(units=32, activation=relu),
        Dense(units=10, activation=softmax)
    ])

    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    return model


def evaluate_model(model, model_name, x, y, verbose=0):
    test_loss, test_accuracy = model.evaluate(x, y, verbose=verbose)
    print(f'Evaluating model {model_name}')
    print(f'  * Val Loss: {np.round(test_loss, 3)}')
    print(f'  * Val Accuracy: {np.round(test_accuracy, 3)}')


train = loadmat('./assets/train_32x32.mat')
test = loadmat('./assets/test_32x32.mat')

X_train = train['X']
y_train = train['y']
X_test = test['X']
y_test = test['y']

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

# Move the samples axis as first
X_train = np.moveaxis(X_train, -1, 0)
X_test = np.moveaxis(X_test, -1, 0)

X_train_grayscale = np.mean(X_train, axis=3, keepdims=True)

print('X train shape: ', X_train.shape)

print('three first values: \n', X_train[0, :3, :3, :])

print('Avergate: \n', np.round(X_train_grayscale[0, :3, :3, 0], 2))

print(np.round((33 + 28 + 40) / 3, 2))
print(np.round((30 + 39 + 41) / 3, 2))
print(np.round((38 + 35 + 38) / 3, 2))

manual_avgs = [[], [], []]

for i in range(3):
    for j in range(3):
        print('Vals to sum: ', X_train[0, i, j, :])
        manual_avgs[i].append(np.sum(X_train[0, i, j, :]) / 3)

print(np.array(manual_avgs))

print(' first: ', X_train_grayscale)

X_test_grayscale = np.mean(X_test, axis=3, keepdims=True) / 255.

mlp_model = get_mlp_model()
mlp_model.summary()
mlp_model.load_weights(tf.train.latest_checkpoint('./best_accurate_model'))

evaluate_model(mlp_model, model_name='MLP Model', x=X_test_grayscale, y=y_test)

print(X_test_grayscale.shape)

for i in range(5):
    idx = np.random.choice(range(X_test.shape[-1]))
    ax[i].set_axis_off()
    ax[i].imshow(X_test[:, :, :, idx])
    ax[i].set_title(y_test[idx])

plt.show()
