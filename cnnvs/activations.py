import tensorflow as tf


def prelu(x):
    name = x.name.partition('/')[0] + '_prelu'
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

