from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import tensorflow_hub as hub

model = Sequential(
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"))

model.build([None, 160, 160, 3])  # Batch input shape.

model.summary()

lemon_img = image.load_img('./assets/lemon.jpg', target_size=(160, 160))
viaduct_img = image.load_img('./assets/viaduct.jpg', target_size=(160, 160))
water_tower_img = image.load_img('./assets/water_tower.jpg', target_size=(160, 160))

with open('./assets/imagenet_categories.txt', 'r') as f:
    categories = f.read().splitlines()


def get_top_5_predictions(img):
    X_input = image.img_to_array(img)[np.newaxis, ...] / 255.
    predictions = model.predict(X_input)
    top_predictions = pd.DataFrame(columns=['prediction'],
                                   index=np.arange(15) + 1)
    sort_idx = np.argsort(-predictions[0])
    for i in range(15):
        i_pred = categories[sort_idx[i]]
        top_predictions.loc[i + 1, 'prediction'] = i_pred

    return top_predictions

print(get_top_5_predictions(lemon_img))
print(get_top_5_predictions(viaduct_img))
print(get_top_5_predictions(water_tower_img))
