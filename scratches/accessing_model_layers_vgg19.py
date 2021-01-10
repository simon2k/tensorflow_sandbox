import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image

vgg_model = VGG19()
vgg_model.summary()

vgg_input = vgg_model.inputs
vgg_layers = vgg_model.layers

layer_output = [layer.output for layer in vgg_layers]
features = Model(inputs=vgg_input, outputs=layer_output)

plot_model(features, to_file='my_vgg_model.png')

random_image = np.random.random(size=(1, 224, 224, 3)).astype('float32')

extracted_features = features(random_image)

print('extracted_features:\n', len(extracted_features))
print('extracted_features shape:\n', extracted_features[0].shape)

cat_image = image.load_img('./assets/cat.jpeg', target_size=(224, 224))

x_input = image.img_to_array(cat_image)
x_input = x_input[np.newaxis, ...]
x_input = preprocess_input(x_input)

extracted_features = features(x_input)

# f1 = extracted_features[0]
# print('f1 shape: ', f1.shape)
#
# imgs = f1[0, :, :]
# plt.figure(figsize=(15, 15))
# for n in range(3):
#     ax = plt.subplot(1, 3, n+1)
#     plt.imshow(imgs[:, :, n])
#     plt.axis('off')
#
# plt.subplots_adjust(wspace=0.01, hspace=0.01)
#
# f2 = extracted_features[1]
# print('f2 shape: ', f2.shape)
#
# imgs = f2[0, :, :]
# plt.figure(figsize=(15, 15))
# for n in range(16):
#     ax = plt.subplot(4, 4, n+1)
#     plt.imshow(imgs[:, :, n])
#     plt.axis('off')
#
# plt.subplots_adjust(wspace=0.01, hspace=0.01)
# plt.show()
#
# f3 = extracted_features[2]
# print('f3 shape: ', f3.shape)
#
# imgs = f3[0, :, :]
# plt.figure(figsize=(15, 15))
# for n in range(16):
#     ax = plt.subplot(4, 4, n+1)
#     plt.imshow(imgs[:, :, n])
#     plt.axis('off')
#
# plt.subplots_adjust(wspace=0.01, hspace=0.01)
# plt.show()
#
# imgs = f3[0, :, :]
# plt.figure(figsize=(15, 15))
# for n in range(16):
#     ax = plt.subplot(4, 4, n+1)
#     plt.imshow(imgs[:, :, n])
#     plt.axis('off')
#
# plt.subplots_adjust(wspace=0.01, hspace=0.01)
# plt.show()

outputs_layer = features.get_layer('block1_pool')
extracted_features_block1_pool = Model(inputs=features.input, outputs=outputs_layer.output)
block1_pool_features = extracted_features_block1_pool.predict(x_input)

print(block1_pool_features[0].shape)

imgs = block1_pool_features[0, :, :]
plt.figure(figsize=(15, 15))
for n in range(64):
    ax = plt.subplot(8, 8, n+1)
    plt.imshow(imgs[:, :, n])
    plt.axis('off')

plt.subplots_adjust(wspace=0.01, hspace=0.01)

outputs_layer = features.get_layer('block5_conv4')
extracted_features_block5_conv4 = Model(inputs=features.input, outputs=outputs_layer.output)
block5_conv4_feats = extracted_features_block5_conv4.predict(x_input)

print(block5_conv4_feats[0].shape)

imgs = block5_conv4_feats[0, :, :]
plt.figure(figsize=(15, 15))
for n in range(64):
    ax = plt.subplot(8, 8, n+1)
    plt.imshow(imgs[:, :, n])
    plt.axis('off')

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()
