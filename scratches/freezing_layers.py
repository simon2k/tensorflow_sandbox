import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import random_uniform, ones, lecun_uniform, zeros
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


def get_weights(model):
    return [e.weights[0].numpy() for e in model.layers]


def get_biases(model):
    return [e.bias.numpy() for e in model.layers]


def plt_delta_weights(W1_layers, W0_layers, b1_layers, b0_layers):
    plt.figure(figsize=(8, 8))

    for n in range(3):
        delta_1 = W1_layers[n] - W0_layers[n]
        b1_norm = np.linalg.norm(b1_layers[n] - b0_layers[n])
        print(f'Layer {n} bias variation: {b1_norm}')

        ax = plt.subplot(1, 3, n + 1)
        plt.imshow(delta_1)
        plt.title(f'Layer {n}')
        plt.axis('off')

    plt.colorbar()
    plt.suptitle('Weight matrices variation')

    plt.show()


model = Sequential([
    Dense(units=4,
          input_shape=(4,),
          activation=relu,
          trainable=False,  # to freeze the layer
          kernel_initializer=random_uniform(),
          bias_initializer=ones()),
    Dense(units=2, activation=relu, kernel_initializer=lecun_uniform(), bias_initializer=zeros()),
    Dense(units=4, activation=softmax)
])

model.summary()

W0_layers = get_weights(model)
b0_layers = get_biases(model)

X_train = np.random.random((100, 4))
y_train = X_train

X_test = np.random.random((20, 4))
y_test = X_test

model.get_layer('dense_1').trainable = False  # Disable the training of the second layer

model.compile(optimizer=Adam(),
              loss=MeanSquaredError(),
              metrics=['accuracy'])

model.fit(x=X_train, y=y_train, epochs=100, validation_data=(X_test, y_test))

W1_layers = get_weights(model)
b1_layers = get_biases(model)

plt_delta_weights(W1_layers, W0_layers, b1_layers, b0_layers)

trainable_vars = model.trainable_variables
non_trainable_vars = model.non_trainable_variables

print('trainable vars:\n', trainable_vars)
print('non trainable vars:\n', non_trainable_vars)

print('# trainable vars:\n', len(trainable_vars))
print('# non trainable vars:\n', len(non_trainable_vars))
