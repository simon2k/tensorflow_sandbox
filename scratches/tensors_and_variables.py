import numpy as np
import tensorflow as tf
from tensorflow import Tensor, Variable, constant

strings = Variable(['something, something'], dtype=tf.string)
floats = Variable([3.14, 2.72], dtype=tf.float64)
ints = tf.Variable([1, 2, 3], tf.int32)
complexs = tf.Variable([-1.2j, 1.3 - 4j], tf.complex64)

# initialize var with shape
print(tf.Variable(tf.constant(4.2, shape=(3, 3))))

# Change vars
a = Variable(0.)
b = a + 1
print(a)
print(b)

print(type(b))

a.assign_add(1)
print(a)

a.assign_sub(4)
print(a)

# Tensors

# Constant tensor

c = tf.constant([[1, 1, 1], [2, 3, 3], [4, 4, 4]])
print(c.dtype, c.shape)

print(c.numpy())

d = tf.constant([[1, 1, 1], [2, 3, 3], [4, 4, 4]], dtype=tf.float32)
print(d)

# Coefficients - split list into different shaes
coeff = np.arange(16)

shape1 = [8, 2]
shape2 = [4, 4]
shape3 = [2, 2, 2, 2]

e1 = tf.constant(coeff, shape=shape1)
e2 = tf.constant(coeff, shape=shape2)
e3 = tf.constant(coeff, shape=shape3)

print(e1, '\n', e2, '\n', e3)

# Useful ops
t = constant(np.arange(100), shape=(10, 2, 5))

# Rank
print('rank: ', tf.rank(t))

# Reshape
t2 = tf.reshape(t, (100, 1))
print('new shape: ', t2.shape)

# Ones and zeros
ones = tf.ones(shape=(2, 3))
zeros = tf.zeros(shape=(4, 2))
eye = tf.eye(5)
my_tensor = tf.constant(4., shape=(2, 3))

print('ones: ', ones)
print('zeros: ', zeros)
print('eye: ', eye)
print('my_tensor: ', my_tensor)

# Concatenate two tensors:
t1 = tf.ones(shape=(2, 2))
t2 = tf.zeros(shape=(2, 2))
c1 = tf.concat([t1, t2], axis=0)
c2 = tf.concat([t1, t2], axis=1)

print('c1\n', c1)
print('c2\n', c2)

# Expanding rank of a tensor
t = tf.constant(np.arange(24), shape=(3, 2, 4))

t1 = tf.expand_dims(t, 0)
t2 = tf.expand_dims(t, 1)
t3 = tf.expand_dims(t, 2)

print('t1 shape: \n', t1.shape)
print('t2 shape: \n', t2.shape)
print('t3 shape: \n', t3.shape)

t1 = tf.squeeze(t1, axis=0)
t2 = tf.squeeze(t2, axis=1)
t3 = tf.squeeze(t3, axis=2)

print('t1 shape sqzd: \n', t1.shape)
print('t2 shape sqzd: \n', t2.shape)
print('t3 shape sqzd: \n', t3.shape)

# Slicing
g = tf.constant(np.arange(8))
print('sliced g: ', g[3:6])

# Math
c = tf.constant([[1., 2.],
                 [3., 4.]])

d = tf.constant([[2., 4.],
                 [6., 8.]])

print('multiplication: ', tf.matmul(c, d))

# element wise ops
print('c * d: \n', c * d)
print('c / d: \n', c / d)
print('c + d: \n', c + d)
print('c - d: \n', c - d)

# Absolute vals
print('abs from c - d: \n', tf.abs(c - d))
print('pow from c, d: \n', tf.pow(c, d))

# Random
r = tf.random.normal(shape=(2, 3), mean=0, stddev=0.3)
print('r: \n', r)

r_complex = tf.cast(r, dtype=tf.complex128)
print('r_complex: \n', r_complex)

u = tf.random.uniform(shape=(3, 3), minval=-1, maxval=5, dtype='float32')
print('u: \n', u)

p = tf.random.poisson(shape=(3, 2), lam=5)
print('p: \n', p)

print('pow r_complex: \n', tf.pow(r_complex, r_complex))

print('abs * 2 of u: \n', tf.abs(u) * 2)


compl = tf.constant([[3.2 + 1.5j], [2.8 - 3.8j]])
print('real: \n', tf.math.real(compl))
print('imag: \n', tf.math.imag(compl))
print()

a1 = tf.random.poisson(shape=(1, 2), dtype=tf.int32, lam=3)
a2 = tf.random.poisson(shape=(1, 2), dtype=tf.int32, lam=3)
print('a1: ', a1)
print('a2: ', a2)
print('less: ', tf.math.less(a1, a2))
print('greater: ', tf.math.greater(a1, a2))
