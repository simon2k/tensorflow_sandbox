from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu, elu
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np

diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
target = diabetes_dataset['target']

target = (target - target.mean(axis=0)) / target.std()

train_data, test_data, train_targets, test_targets = train_test_split(data, target, test_size=0.1)

model = Sequential([
    Dense(units=128, activation=relu, input_shape=(train_data.shape[1],)),
    Dense(units=64, activation=relu),
    BatchNormalization(),
    Dense(units=32, activation=relu),
    Dense(units=32, activation=relu),
    Dense(units=1)
])

model.compile(optimizer=SGD(), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])


class MetricLossCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch % 2 == 0:
            print(f'[Train] After batch {batch} - loss {logs["loss"]}')

    def on_test_batch_end(self, batch, logs=None):
        print(f'[Test] After batch {batch} - loss {logs["loss"]}')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch} avg loss is: {logs["loss"]} MAE is {logs["mean_absolute_error"]}')

    def on_predict_batch_end(self, batch, logs=None):
        print(f"Finished predicting batch {batch}")


metric_loss_callback = MetricLossCallback()

history = model.fit(train_data, train_targets, epochs=100, verbose=False, batch_size=128,
                    callbacks=[metric_loss_callback])

model_evaluation = model.evaluate(test_data, test_targets, callbacks=[metric_loss_callback], verbose=False)

print('\n\nPREDICTION TIME\n\n')
model_prediction = model.predict(test_data[0][np.newaxis, ...], callbacks=[metric_loss_callback])
print(model_prediction, test_targets[0])

model_prediction = model.predict(test_data[2][np.newaxis, ...], callbacks=[metric_loss_callback])
print(model_prediction, test_targets[2])

print('-----------------------')

lr_schedule = [
    (4, .03),
    (7, .02),
    (11, .005),
    (15, .007)
]


def get_new_epoch_lr(epoch, lr):
    epoch_in_schedule = [i for i in range(len(lr_schedule)) if lr_schedule[i][0] == int(epoch)]

    if len(epoch_in_schedule) > 0:
        return lr_schedule[epoch_in_schedule[0]][1]
    else:
        return lr


class LearningRateScheduler(Callback):
    def __init__(self, new_lr):
        super(LearningRateScheduler, self).__init__()
        self.new_lr = new_lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Error: Selected optimizer has no learning rate to configure')

        curr_rate = float(K.get_value(self.model.optimizer.lr))
        schedule_rate = self.new_lr(epoch, curr_rate)

        K.set_value(self.model.optimizer.lr, schedule_rate)
        print(f'Learning rate for epoch {epoch} is set to {schedule_rate}')


history = model.fit(train_data, train_targets, epochs=100, verbose=False, batch_size=128,
                    callbacks=[LearningRateScheduler(get_new_epoch_lr)])

print(history.history)
