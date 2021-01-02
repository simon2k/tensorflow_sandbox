from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

model = ResNet50(weights='imagenet', include_top=True)

model.summary()

lemon_img = image.load_img('./assets/lemon.jpg', target_size=(224, 224))
viaduct_img = image.load_img('./assets/viaduct.jpg', target_size=(224, 224))
water_tower_img = image.load_img('./assets/water_tower.jpg', target_size=(224, 224))


def get_top_5_predictions(img):
    X_input = image.img_to_array(img)[np.newaxis, ...]
    X_input = preprocess_input(X_input)

    predictions = decode_predictions(model.predict(X_input), top=15)
    top_predictions = pd.DataFrame(columns=['prediction', 'probability'],
                                   index=np.arange(15) + 1)
    for i in range(15):
        top_predictions.loc[i + 1, 'prediction'] = predictions[0][i][1]
        top_predictions.loc[i + 1, 'probability'] = predictions[0][i][2]

    return top_predictions


print(get_top_5_predictions(lemon_img))
print(get_top_5_predictions(viaduct_img))
print(get_top_5_predictions(water_tower_img))

# img_input = image.img_to_array(img_input)
# img_input = preprocess_input(img_input[np.newaxis, ...])
#
# print(img_input.shape)
#
# predictions = model.predict(img_input)
# decoded_predictions = decode_predictions(predictions, top=3)[0]
#
# print(decoded_predictions)
