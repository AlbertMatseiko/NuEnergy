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


def shifted_relu(x, t=1, a=1):
    return tf.keras.activations.relu(a*x+t)

def my_softplus(x, a=0.3, b=2.):
    return a*tf.keras.activations.softplus(b*x)


WideTanh = tf.keras.layers.Activation(wide_tanh)
WideSigmoid = tf.keras.layers.Activation(wide_sigmoid)

ShiftedRelu = tf.keras.layers.Activation(shifted_relu)
MySoftplus = tf.keras.layers.Activation(my_softplus)
