import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input

input_a = Input(shape=(128, 128, 3), name='input_a')
input_b = Input(shape=(64, 64, 3), name='input_b')

conv = Conv2D(filters=32, kernel_size=(6, 6), padding='same')
conv_out_input_a = conv(input_a)
print('conv: ', conv)
print('conv_out_input_a: ', conv_out_input_a)
print('type conv_out_input_a: ', type(conv_out_input_a))

conv_out_input_b = conv(input_b)
print('conv: ', conv)
print('conv_out_input_b: ', conv_out_input_b)
print('type conv_out_input_b: ', type(conv_out_input_b))

print('conv input: ', conv.input)
print('conv output: ', conv.output)

# print(conv.input_shape)
# AttributeError: The layer "conv2d has multiple inbound nodes, with different input shapes. Hence the notion
# of "input shape" is ill-defined for the layer. Use `get_input_shape_at(node_index)` instead.
# print(conv.output_shape)
# AttributeError: The layer "conv2d has multiple inbound nodes, with different input shapes. Hence the notion
# of "input shape" is ill-defined for the layer. Use `get_input_shape_at(node_index)` instead.

print('conv input shape @ 0: ', conv.get_input_shape_at(0))
print('conv input shape @ 1: ', conv.get_input_shape_at(1))
print('conv output shape @ 0: ', conv.get_input_shape_at(0))
print('conv output shape @ 1: ', conv.get_input_shape_at(1))

print('conv input @ 0: ', conv.get_input_at(0))
print('conv input @ 1: ', conv.get_input_at(1))
print('conv output @ 0: ', conv.get_output_at(0))
print('conv output @ 1: ', conv.get_output_at(1))
