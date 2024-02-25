import tensorflow as tf


# for my energy
def wide_tanh(x, a=4.):
    b = 1 / a
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    x = tf.multiply(b, x)
    activation = tf.multiply(a, tf.keras.activations.tanh(x))
    return activation


# for sigma
def wide_sigmoid(x, a=10.):
    b = 1 / a
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    x = tf.multiply(b, x)
    activation = tf.multiply(a, tf.keras.activations.sigmoid(x))
    return activation


def shifted_relu(x, t=2., a=0.5):
    return tf.keras.activations.relu(a*x+t)


WideTanh = tf.keras.layers.Activation(wide_tanh)
WideSigmoid = tf.keras.layers.Activation(wide_sigmoid)

ShiftedRelu = tf.keras.layers.Activation(shifted_relu)
