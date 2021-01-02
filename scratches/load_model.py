from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
import json

model = load_model('./training_assets_entire_model/model-10')

model.summary()

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = X_train / 255.
# X_test = X_test / 255.


def get_test_accuracy(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(x=X_test, y=y_test, verbose=0)
    print(f'Loss: {np.round(test_loss, 3)} accuracy: {np.round(test_accuracy, 3)}')


# get_test_accuracy(model, X_test, y_test)
#
# save_model_checkpoint_callback = ModelCheckpoint(
#     filepath='saved_model/model-{epoch}',
#     save_weights_only=False,
#     mode='max',
#     save_freq='epoch',
#     save_best_only=True,
#     verbose=1
# )
#
# model.fit(
#     x=X_train,
#     y=y_train,
#     epochs=3,
#     batch_size=16,
#     callbacks=[save_model_checkpoint_callback],
#     validation_data=(X_test, y_test)
# )

# get_test_accuracy(model, X_test, y_test)

model_config = model.get_config()
# print(model_config)

new_model = Sequential.from_config(model_config)
new_model.summary()

model_json_config = model.to_json()
print(model_json_config)

with open('model-config.json', 'w') as f:
    json.dump(model_json_config, f)

with open('model-config.json', 'r') as f:
    new_json_model_config = json.load(f)

new_model_from_json = model_from_json(new_json_model_config)

print('new_model_from_json.summary')
new_model_from_json.summary()
