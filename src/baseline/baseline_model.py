# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""

import tensorflow as tf


def build_baseline_model(input_tf):
    """
    Build the baseline model
    Args:
        input_tf (Tensor):

    Returns:
        Output tensor
    """
    # First layer
    with tf.variable_scope('Conv_{}'.format(1)):
        x = tf.layers.conv2d(inputs=input_tf,
                             filters=64,
                             kernel_size=256,
                             strides=(1, 1),
                             activation=None,
                             kernel_initializer=None,
                             name='conv2d')
    with tf.variable_scope('Conv_{}'.format(2)):
        x = tf.layers.conv2d(inputs=x,
                             filters=128,
                             kernel_size=128,
                             strides=(1, 1),
                             activation=None,
                             kernel_initializer=None,
                             name='conv2d')
    with tf.variable_scope('Conv_{}'.format(3)):
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=64,
                             strides=(1, 1),
                             activation=None,
                             kernel_initializer=None,
                             name='conv2d')
    # Identical layers
    for i in range(4, 8):
        with tf.variable_scope('Conv_{}'.format(i)):
            x = tf.layers.conv2d(inputs=x,
                                 filters=512,
                                 kernel_size=32,
                                 dilation_rate=(2, 2) if i in {5, 6} else (1, 1),
                                 strides=(1, 1),
                                 activation=None,
                                 kernel_initializer=None,
                                 name='conv2d')
    with tf.variable_scope('Conv_{}'.format(8)):
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=64,
                             strides=(1, 1),
                             activation=None,
                             kernel_initializer=None,
                             name='conv2d')
    return x
