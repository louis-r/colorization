# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""

import tensorflow as tf


def return_padded(x, pad=1):
    """
    Pad input. See tensorflow doc.
    Args:
        x ():
        pad ():

    Returns:

    """
    # Padding of 1
    paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    x_padded = tf.pad(tensor=x, paddings=paddings)
    return x_padded


def build_baseline_model(input_tf):
    """
    Build the baseline model
    Args:
        input_tf (Tensor):

    Returns:
        Output tensor
    """
    # TODO Complete, review, test. What is param {lr_mult: 0 decay_mult: 0}?
    # Block CNN 1-2
    x = input_tf
    for i in range(1, 2):
        with tf.variable_scope('BCNN_{}'.format(i)):
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=2 ** (i + 5),
                                 kernel_size=3,
                                 strides=1,
                                 activation=tf.nn.relu,
                                 name='conv_{}_1'.format(i))
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=2 ** (i + 5),
                                 kernel_size=3,
                                 strides=2,
                                 activation=tf.nn.relu,
                                 name='conv_{}_2'.format(i))
            x = tf.layers.batch_normalization(inputs=x,
                                              name='conv_{}_batchnorm'.format(i))
    # Block CNN 3
    with tf.variable_scope('BCNN_{}'.format(3)):
        # Padding of 1
        x = return_padded(x=x)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             name='conv_{}_1'.format(3))
        # Padding of 1
        x = return_padded(x=x)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             name='conv_{}_2'.format(3))
        # Padding of 1
        x = return_padded(x=x)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=3,
                             strides=2,
                             activation=tf.nn.relu,
                             name='conv_{}_3'.format(3))
        x = tf.layers.batch_normalization(inputs=x,
                                          name='conv_{}_batchnorm'.format(3))

    # Block CNN 4
    with tf.variable_scope('BCNN_{}'.format(4)):
        # Padding of 1
        x = return_padded(x=x)
        x = tf.layers.conv2d(inputs=x,
                             filters=512,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_1'.format(4))
        # Padding of 1
        x = return_padded(x=x)
        x = tf.layers.conv2d(inputs=x,
                             filters=512,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_2'.format(4))
        # Padding of 1
        x = return_padded(x=x)
        x = tf.layers.conv2d(inputs=x,
                             filters=512,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_3'.format(4))

        x = tf.layers.batch_normalization(inputs=x,
                                          name='conv_{}_batchnorm'.format(4))

    # Block CNN 5-6
    for i in range(5, 7):
        with tf.variable_scope('BCNN_{}'.format(i)):
            # Padding of 2
            x = return_padded(x=x, pad=2)
            x = tf.layers.conv2d(inputs=x,
                                 filters=512,
                                 kernel_size=3,
                                 strides=1,
                                 activation=tf.nn.relu,
                                 dilation_rate=2,
                                 name='conv_{}_1'.format(i))
            # Padding of 2
            x = return_padded(x=x, pad=2)
            x = tf.layers.conv2d(inputs=x,
                                 filters=512,
                                 kernel_size=3,
                                 strides=1,
                                 activation=tf.nn.relu,
                                 dilation_rate=2,
                                 name='conv_{}_2'.format(i))
            # Padding of 2
            x = return_padded(x=x, pad=2)
            x = tf.layers.conv2d(inputs=x,
                                 filters=512,
                                 kernel_size=3,
                                 strides=1,
                                 activation=tf.nn.relu,
                                 dilation_rate=2,
                                 name='conv_{}_3'.format(i))
            x = tf.layers.batch_normalization(inputs=x,
                                              name='conv_{}_batchnorm'.format(i))
    # Block CNN 7
    with tf.variable_scope('BCNN_{}'.format(7)):
        # Padding of 1
        x = return_padded(x=x, pad=1)
        x = tf.layers.conv2d(inputs=x,
                             filters=512,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             kernel_initializer=None,
                             name='conv_{}_1'.format(7))
        # Padding of 1
        x = return_padded(x=x, pad=1)
        x = tf.layers.conv2d(inputs=x,
                             filters=512,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             kernel_initializer=None,
                             name='conv_{}_2'.format(7))
        # Padding of 1
        x = return_padded(x=x, pad=1)
        x = tf.layers.conv2d(inputs=x,
                             filters=512,
                             kernel_size=3,
                             strides=1,
                             activation=tf.nn.relu,
                             kernel_initializer=None,
                             name='conv_{}_3'.format(7))
        x = tf.layers.batch_normalization(inputs=x,
                                          name='conv_{}_batchnorm'.format(7))
    # Block CNN 8
    # Deconvo layer
    with tf.variable_scope('BCNN_{}'.format(8)):
        # Padding of 1
        x = return_padded(x=x, pad=1)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=3,
                             strides=2,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_1'.format(8))
        # Padding of 1
        x = return_padded(x=x, pad=1)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=3,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_2'.format(8))
        # Padding of 1
        x = return_padded(x=x, pad=1)
        x = tf.layers.conv2d(inputs=x,
                             filters=256,
                             kernel_size=3,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_3'.format(8))
    # Block CNN 9
    # TODO Not finished
    with tf.variable_scope('BCNN_{}'.format(9)):
        x = tf.layers.conv2d(inputs=x,
                             filters=313,
                             kernel_size=1,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_1'.format(9))

        x = tf.layers.conv2d(inputs=x,
                             filters=2,
                             kernel_size=1,
                             activation=tf.nn.relu,
                             dilation_rate=1,
                             name='conv_{}_2'.format(9))
        # Product layer
        # TODO Implement product layer
        # Tmp
        # x = tf.reshape(x, [-1, 28 * 28 * 2])
        # Softmax layer
        # softmax = tf.nn.softmax(logits=x,
        #                         name='softmax_{}'.format(9))

    return x
