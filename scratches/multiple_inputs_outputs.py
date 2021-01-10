import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy

"""
 Inputs:
 * 0 Temperature 35 - 42 deg C
 * 1 Occurrence of nausea: bool
 * 2 Lumbar pain: bool
 * 3 Urine pushing (need to urinate): bool
 * 4 Micturition pains: bool
 * 5 Burning of urethrea, itch, swelling of urethrea outlet: bool
 
 Output:
 * 6 Decision 1: Inflammation of urinary bladder: bool
 * 7 Decision 2: Nephritis of renal pelvis origin: bool
"""

diagnostic_data = pd.read_csv('./assets/diagnosis.csv')
dataset = diagnostic_data.values
print(diagnostic_data.describe())

X_train, X_test, y_train, y_test = train_test_split(dataset[:, :6], dataset[:, 6:], test_size=.33)

temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train = np.transpose(X_train)
temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test = np.transpose(X_test)

inflam_train, nephr_train = y_train[:, 0], y_train[:, 1]
inflam_test, nephr_test = y_test[:, 0], y_test[:, 1]

shape_inputs = (1,)
temp = Input(shape=shape_inputs, name='temp')
nocc = Input(shape=shape_inputs, name='nocc')
lumbp = Input(shape=shape_inputs, name='lumbp')
up = Input(shape=shape_inputs, name='up')
mict = Input(shape=shape_inputs, name='mict')
bis = Input(shape=shape_inputs, name='bis')

inputs = [
    temp,
    nocc,
    lumbp,
    up,
    mict,
    bis
]

h = concatenate(inputs=inputs)

inflam_pred = Dense(units=1, activation=sigmoid, name='inflam')(h)
nephr_pred = Dense(units=1, activation=sigmoid, name='nephr')(h)

outputs = [inflam_pred, nephr_pred]

model = Model(inputs=inputs, outputs=outputs)

plot_model(model, 'model.png', show_shapes=True)

model.compile(
    optimizer=RMSprop(learning_rate=1e-3),
    loss={'inflam': BinaryCrossentropy(), 'nephr': BinaryCrossentropy()},
    metrics={'inflam': ['accuracy'], 'nephr': ['accuracy']},
    loss_weights=[1., .2]
)

inputs_train = {
    'temp': temp_train,
    'nocc': nocc_train,
    'lumbp': lumbp_train,
    'up': up_train,
    'mict': mict_train,
    'bis': bis_train
}

outputs_train = {
    'inflam': inflam_train,
    'nephr': nephr_train
}

model.summary()

history = model.fit(x=inputs_train, y=outputs_train, epochs=1000).history

accuracy_keys = [k for k in history.keys() if k in ('inflam_accuracy', 'nephr_accuracy')]
loss_keys = [k for k in history.keys() if not k in accuracy_keys]

for k, v in history.items():
    if k in accuracy_keys:
        plt.figure(1)
    else:
        plt.figure(2)
    plt.plot(v)

plt.figure(1)
plt.title('Accuracy over Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(accuracy_keys, loc='lower right')

plt.figure(2)
plt.title('Loss over Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loss_keys, loc='upper right')

plt.show()

inputs_test = {
    'temp': temp_test,
    'nocc': nocc_test,
    'lumbp': lumbp_test,
    'up': up_test,
    'mict': mict_test,
    'bis': bis_test
}

outputs_test = {
    'inflam': inflam_test,
    'nephr': nephr_test
}

# loss: 0.1705 - inflam_loss: 0.1183 - nephr_loss: 0.2611 - inflam_accuracy: 1.0000 - nephr_accuracy: 1.0000
# [test_loss, test_inflam_loss, test_nephr_loss, test_inflam_acc, test_nephr_acc] = model.evaluate(x=inputs_test, y=outputs_test)
(test_loss,
 test_inflam_loss,
 test_nephr_loss,
 test_inflam_acc,
 test_nephr_acc) = model.evaluate(x=inputs_test, y=outputs_test, verbose=1)
# print(x)

# print(test_inflam_loss, test_nephr_loss, test_inflam_acc, test_nephr_acc)

# print(f'\n* Test loss: {np.round(test_loss, 3)}'
#       f'\n* Test accuracy: {np.round(test_accuracy, 3)}')
