# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
# pylint: disable=C
import os
import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from baseline_model import build_baseline_model_v2

# from tensorflow.examples.tutorials.mnist import import input_data

tf.app.flags.DEFINE_string('train_dir', 'runs/',
                           """Output dir for tensorflow summaries.""")
tf.app.flags.DEFINE_string('prefix', 'prefix',
                           """Model name prefix.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-2,
                          """Learning rate.""")
tf.app.flags.DEFINE_integer('n_steps', 10,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Batch size.""")

FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir
prefix = FLAGS.prefix
learning_rate = FLAGS.learning_rate
n_steps = FLAGS.n_steps
batch_size = FLAGS.batch_size

model_name = '{}_{}_{}_{}'.format(prefix,
                                  learning_rate,
                                  n_steps,
                                  batch_size)

# Do not specify the size of the training batch
L_tf = tf.placeholder(tf.float32, [None, 224, 224, 1], name='L_tf')
T_recip_tf = tf.placeholder(tf.float32, [None, 56, 56, 313], name='T_recip_tf')

tf.summary.image('L_tf', L_tf, max_outputs=3)

# TODO incorrect output shape for now
# Predicted values
logits, Z_pred, y_pred = build_baseline_model_v2(input_tf=L_tf)

# Display image
# tf.summary.image('y_pred_a', tf.expand_dims(input=Z_pred[:, :, :, 0], axis=3), max_outputs=3)
# tf.summary.image('y_pred_b', tf.expand_dims(input=Z_pred[:, :, :, 1], axis=3), max_outputs=3)

H_out, W_out = y_pred.get_shape().as_list()[1:3]

# Ground truth
y_true = tf.placeholder(tf.float32, [None, H_out, W_out, 2], name='y_true')
# Z_true hard-encoding of the ground_truth color
# Z_true =

print('L_tf shape = {}'.format(L_tf.get_shape().as_list()))
print('T_recip_tf shape = {}'.format(T_recip_tf.get_shape().as_list()))
print('logits shape = {}'.format(logits.get_shape().as_list()))
print('Z_pred shape = {}'.format(Z_pred.get_shape().as_list()))
print('y_pred shape = {}'.format(y_pred.get_shape().as_list()))

# Metrics
with tf.name_scope("loss"):
    loss = tf.reduce_mean(2 * tf.nn.l2_loss(t=y_true - y_pred),
                          name='reduced_l2_loss')
    loss_summary_tf = tf.summary.scalar("loss", loss)

global_step = tf.Variable(0, trainable=False, name='global_step')

# Optimizer
with tf.name_scope("optimizer"):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                           global_step=global_step)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

train_summaries_tf = tf.summary.merge_all()
test_summaries_tf = tf.summary.merge([loss_summary_tf])

train_write_path = train_dir + model_name + '/train'
test_write_path = train_dir + model_name + '/test'

print('Saving at {}'.format(train_dir + model_name))
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
os.mkdir(train_dir + model_name)
os.mkdir(train_write_path)
os.mkdir(test_write_path)

train_writer = tf.summary.FileWriter(train_write_path)
test_writer = tf.summary.FileWriter(test_write_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # TensorBoard operations
    train_writer.add_graph(sess.graph)

    for step in range(n_steps):
        # Train
        batch_xs, batch_ys = np.random.rand(batch_size, 224, 224, 1), np.random.rand(batch_size, H_out, W_out, 2)

        _, train_loss_val, s = sess.run(
            [train_step, loss, train_summaries_tf],
            feed_dict={
                L_tf: batch_xs,
                y_true: batch_ys
            })

        train_writer.add_summary(s, step)

        print('TRAIN Step = %04d\tloss = %.4f' % (step + 1,
                                                  train_loss_val))
        # Cross validate
        test_batch_xs, test_batch_ys = np.random.rand(batch_size, 224, 224, 1), np.random.rand(batch_size, H_out, W_out,
                                                                                               2)
        test_loss_val, s = sess.run(
            [loss, test_summaries_tf],
            feed_dict={
                L_tf: test_batch_xs,
                y_true: test_batch_ys
            })
        print('TEST Step = %04d\tloss = %.4f' % (step + 1,
                                                 test_loss_val))

        test_writer.add_summary(s, step)

print("Run the command line:\n"
      "--> tensorboard --logdir={} "
      "\nThen open http://0.0.0.0:6006/ into your web browser".format(train_dir))
