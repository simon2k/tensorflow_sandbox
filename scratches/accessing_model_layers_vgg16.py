import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

vgg16_model = VGG16()
vgg16_model.summary()

layer_outputs = [layer.output for layer in vgg16_model.layers]

features = Model(inputs=vgg16_model.inputs, outputs=layer_outputs)

features.summary()

plot_model(features, to_file='vgg_16_custom_model.png')

# random_image = np.random.random(size=(1, 224, 224, 3)).astype('float32')
random_image = image.load_img(path='./assets/cat.jpeg', target_size=(224, 224))
random_image = image.img_to_array(random_image)[np.newaxis, ...]
x_input = preprocess_input(random_image)

extracted_features = features(x_input)

f1 = extracted_features[0]
print(f1.shape)
processed_imgs = f1[0, :, :]
print(processed_imgs.shape)

plt.figure(figsize=(15, 15))

for n in range(3):
    ax = plt.subplot(1, 3, n+1)
    plt.imshow(processed_imgs[:, :, n])
    plt.axis('off')

f3 = extracted_features[2]
print(f3.shape)
processed_imgs = f3[0, :, :]
print(processed_imgs.shape)

plt.figure(figsize=(15, 15))

for n in range(16):
    ax = plt.subplot(4, 4, n+1)
    plt.imshow(processed_imgs[:, :, n])
    plt.axis('off')

plt.subplots_adjust(wspace=.1, hspace=.1)


b5c3_layer = features.get_layer('block4_conv3')
feats_b5c3_model = Model(inputs=vgg16_model.inputs, outputs=b5c3_layer.output)
feats_b5c3 = feats_b5c3_model.predict(x_input)

plt.figure(figsize=(8, 8))

print(feats_b5c3, feats_b5c3.shape)

for n in range(64):
    ax = plt.subplot(8, 8, n+1)
    plt.imshow(feats_b5c3[0][:, :, n], interpolation='nearest')
    plt.axis('off')

plt.show()
