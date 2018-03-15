# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus


This is the V2 model!
"""

import tensorflow as tf


def return_padded(x, pad=1):
    """
    Pad input. See tensorflow doc.
    Args:
        x (tf.tensor): tensor N, H, W, C to be padded
        pad (int): padding constant

    Returns:

    """
    # Padding of 1
    paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    x_padded = tf.pad(tensor=x, paddings=paddings)
    return x_padded


def build_baseline_model_v2(input_tf, pts_in_hull_tf, batch_size):
    """
    Build the baseline model
    Args:
        batch_size ():
        pts_in_hull_tf ():
        input_tf (Tensor):

    Returns:
        Output tensor
        1x2x56x56 on caffee
        There is a resize operation to match 244x244 input size
    """
    # TODO Complete, review, test. What is param {lr_mult: 0 decay_mult: 0}?
    # Block CNN 1-2
    x = input_tf
    for i in range(1, 3):
        with tf.variable_scope('BCNN_{}'.format(i)):
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=2 ** (i + 5),
                                 kernel_size=3,
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
        for j in range(1, 3):
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=256,
                                 kernel_size=3,
                                 activation=tf.nn.relu,
                                 name='conv_{}_{}'.format(3, j))
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
        for j in range(1, 4):
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=512,
                                 kernel_size=3,
                                 activation=tf.nn.relu,
                                 name='conv_{}_{}'.format(4, j))
        x = tf.layers.batch_normalization(inputs=x,
                                          name='conv_{}_batchnorm'.format(4))

    # Block CNN 5-6
    for i in range(5, 7):
        with tf.variable_scope('BCNN_{}'.format(i)):
            for j in range(1, 4):
                # Padding of 2
                x = return_padded(x=x, pad=2)
                x = tf.layers.conv2d(inputs=x,
                                     filters=512,
                                     kernel_size=3,
                                     activation=tf.nn.relu,
                                     dilation_rate=2,
                                     name='conv_{}_{}'.format(i, j))
            x = tf.layers.batch_normalization(inputs=x,
                                              name='conv_{}_batchnorm'.format(i))
    # Block CNN 7
    with tf.variable_scope('BCNN_{}'.format(7)):
        for j in range(1, 4):
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=512,
                                 kernel_size=3,
                                 activation=tf.nn.relu,
                                 name='conv_{}_{}'.format(7, j))
        x = tf.layers.batch_normalization(inputs=x,
                                          name='conv_{}_batchnorm'.format(7))
    # Block CNN 8
    with tf.variable_scope('BCNN_{}'.format(8)):
        # Padding of 1
        x = return_padded(x=x)
        # x = tf.layers.conv2d_transpose(inputs=x,
        #                                filters=256,
        #                                kernel_size=4,
        #                                strides=2,
        #                                activation=tf.nn.relu,
        #                                name='conv_{}_1_deconvolution'.format(8))
        x = tf.layers.conv2d_transpose(inputs=x,
                                       filters=256,
                                       kernel_size=25,
                                       strides=7,
                                       activation=tf.nn.relu)

        # 34, 34, 512 -> 56, 56, 256
        # filter_deconv = tf.Variable(tf.random_normal([34, 34, 256, 512]))
        # x = tf.nn.conv2d_transpose(value=x,
        #                            filter=filter_deconv,
        #                            output_shape=tf.constant([batch_size, 56, 56, 256]),
        #                            strides=[1, 2, 2, 1],
        #                            padding='VALID')
        # 34, 34, 512 -> 64, 64, 256
        # x = tf.nn.conv2d_transpose(value=x,
        #                            filter=filter_deconv,
        #                            output_shape=tf.constant([batch_size, 64, 64, 256]),
        #                            strides=[1, 2, 2, 1],
        #                            padding='VALID')
        for j in range(2, 4):
            # Padding of 1
            x = return_padded(x=x)
            x = tf.layers.conv2d(inputs=x,
                                 filters=256,
                                 kernel_size=3,
                                 activation=tf.nn.relu,
                                 name='conv_{}_{}'.format(8, j))
    # Block CNN 9
    with tf.variable_scope('BCNN_{}_softmax'.format(9)):
        # Notations from the article
        # conv8_313
        logits = tf.layers.conv2d(inputs=x,
                                  filters=313,
                                  kernel_size=1,
                                  activation=tf.nn.relu,
                                  name='conv_{}_1'.format(9))

        # class8_313_rh
        z = tf.nn.softmax(logits=logits,
                          name='conv_{}_softmax'.format(9))
        # class8_ab
        # class8_ab = tf.layers.conv2d(inputs=x,
        #                              filters=2,
        #                              kernel_size=1,
        #                              name='conv_{}_INCORRECT_LAYER'.format(9))
        # Annealed mean
        class8_ab = tf.nn.conv2d(input=z,
                                 filter=tf.reshape(pts_in_hull_tf,
                                                   [1, 1, 313, 2]),
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='class8_ab')

        # Product layer
        # TODO Implement scale layer
        # Tmp
        # x = tf.reshape(x, [-1, 28 * 28 * 2])
        # Softmax layer
        # softmax = tf.nn.softmax(logits_tf=x,
        #                         name='softmax_{}'.format(9))

    return logits, z, class8_ab
