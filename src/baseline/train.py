# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
# pylint: disable=C
import os
import datetime
import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from baseline_model import build_baseline_model_v2
# noinspection PyUnresolvedReferences
from data_utils import load_data
from image_utils import lab_to_rgb

tf.app.flags.DEFINE_string('train_dir', 'runs/',
                           """Output dir for tensorflow summaries.""")
tf.app.flags.DEFINE_string('prefix', None,
                           """Model name prefix.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-2,
                          """Learning rate.""")
tf.app.flags.DEFINE_integer('n_steps', 1000,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('image_summ_freq', 10,
                            """Frequency of image summaries.""")
tf.app.flags.DEFINE_integer('var_summ_freq', 10,
                            """Frequency of var summaries.""")

# MODEL SPECIFICATION
# Parse flags argument
FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir
prefix = FLAGS.prefix
learning_rate = FLAGS.learning_rate
n_steps = FLAGS.n_steps
batch_size = FLAGS.batch_size
image_summ_freq = FLAGS.image_summ
var_summ_freq = FLAGS.var_summ

model_name = 'lr={}_n_steps={}_batch_size={}'.format(learning_rate,
                                                     n_steps,
                                                     batch_size)
if prefix is not None:
    model_name = '{}_{}'.format(prefix, model_name)

print('Training {} at {}'.format(model_name, datetime.datetime.now()))

H_in, W_in = 256, 256
H_out, W_out = 256, 256
Q = 313

######## DATA ########
# Load the data
L, z_true, y_true = load_data(a_file='../../data/X_lab_a0.npy',
                              b_file='../../data/X_lab_b0.npy',
                              L_file='../../data/X_lab_L0.npy',
                              n_images=batch_size)
# Checks
assert L.shape == (batch_size, H_in, W_in, 1)
assert y_true.shape == (batch_size, H_in, W_in, 2)
assert z_true.shape == (batch_size, H_in, W_in, Q)

######## TENSORFLOW GRAPH ########
L_tf = tf.placeholder(tf.float32, [None, H_in, W_in, 1], name='L_tf')
# 56 shape from caffe_v1.txt
T_recip_tf = tf.placeholder(tf.float32, [None, 56, 56, Q], name='T_recip_tf')
# Cluster centers
pts_in_hull_tf = tf.placeholder(tf.float32, [Q, 2], name='pts_in_hull_tf')

# TODO incorrect output shape for now
# The output shape of y_pred_tf should be (56, 56)
# cf: https://github.com/richzhang/colorization/blob/master/colorization/demo/colorization_demo_v2.ipynb
# Predicted values
logits_tf, Z_pred_tf, y_pred_tf = build_baseline_model_v2(input_tf=L_tf,
                                                          pts_in_hull_tf=pts_in_hull_tf,
                                                          # batch_size=batch_size
                                                          )
# H_out, W_out = y_pred_tf.get_shape().as_list()[1:3]
# assert H_out == W_out == 64, 'Incorrect output shapes'

# Ground truth
y_true_tf = tf.placeholder(tf.float32, [None, H_out, W_out, 2], name='y_true_tf')
# Z_true hard-encoding of the ground_truth color
z_true_tf = tf.placeholder(tf.float32, [None, H_out, W_out, Q], name='z_true_tf')

recolorized_image_tf = tf.concat([L_tf, y_pred_tf], axis=3, name='recolorized_image_tf')
original_image_tf = tf.concat([L_tf, y_true_tf], axis=3, name='original_image_tf')

print('L_tf shape = {}'.format(L_tf.get_shape().as_list()))
print('T_recip_tf shape = {}'.format(T_recip_tf.get_shape().as_list()))
print('logits_tf shape = {}'.format(logits_tf.get_shape().as_list()))
print('Z_pred_tf shape = {}'.format(Z_pred_tf.get_shape().as_list()))
print('y_pred_tf shape = {}'.format(y_pred_tf.get_shape().as_list()))
print('y_true_tf shape = {}'.format(y_true_tf.get_shape().as_list()))
print('z_true_tf shape = {}'.format(z_true_tf.get_shape().as_list()))
print('original_image_tf shape = {}'.format(original_image_tf.get_shape().as_list()))
print('recolorized_image_tf shape = {}'.format(recolorized_image_tf.get_shape().as_list()))

# Metrics
with tf.name_scope("loss"):
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=z_true_tf,
                                           logits=logits_tf)

global_step = tf.Variable(0, trainable=False, name='global_step')

# Optimizer
with tf.name_scope("optimizer"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                global_step=global_step)

######## TENSORFLOW SUMMARIES ########
# Scalars
loss_summary_tf = tf.summary.scalar("loss", loss)
all_scalar_summaries = [loss_summary_tf]

# Images
max_outputs = min(batch_size, 3)  # Number of images per batch to display
L_summary = tf.summary.image(L_tf.name,
                             L_tf,
                             max_outputs=max_outputs)
recolorized_image_summary = tf.summary.image(recolorized_image_tf.name,
                                             lab_to_rgb(recolorized_image_tf),
                                             max_outputs=max_outputs)
original_image_summary = tf.summary.image(original_image_tf.name,
                                          lab_to_rgb(original_image_tf),
                                          max_outputs=max_outputs)
all_image_summaries = [L_summary, recolorized_image_summary, original_image_summary]

# Variables summaries
all_var_summaries = []
for var in tf.trainable_variables():
    all_var_summaries.append(tf.summary.histogram(var.name, var))

scalar_summaries_op = tf.summary.merge(all_scalar_summaries)
var_summaries_op = tf.summary.merge(all_var_summaries)
image_summaries_op = tf.summary.merge(all_image_summaries)

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

    # Add TensorBoard graph
    train_writer.add_graph(sess.graph)

    print('Starting training for n_steps = {}'.format(n_steps))
    for step in range(n_steps):
        # Train
        batch_x = L
        batch_z = z_true
        batch_y = y_true
        train_feed_dict = {
            L_tf: batch_x,
            z_true_tf: batch_z,
            y_true_tf: batch_y,
            pts_in_hull_tf: np.load('pts_in_hull.npy').astype(float),
        }
        _, train_loss_val, scalar_summaries, y_pred_val = sess.run(
            [train_step, loss, scalar_summaries_op, y_pred_tf],
            feed_dict=train_feed_dict)

        train_writer.add_summary(scalar_summaries, step)

        # Also log var summaries
        if step % var_summ_freq == 0:
            var_summaries = sess.run(var_summaries_op,
                                     feed_dict=train_feed_dict)
            train_writer.add_summary(var_summaries, step)

        # Also log image summaries
        if step % image_summ_freq == 0:
            image_summary = sess.run(image_summaries_op,
                                     feed_dict=train_feed_dict)
            train_writer.add_summary(image_summary, step)

        print('TRAIN Step = %04d\tloss = %.6f' % (step + 1,
                                                  train_loss_val))
        # # Cross validate
        # test_batch_x = np.random.rand(batch_size, H_in, W_in, 1)
        # # test_batch_y = np.random.rand(batch_size, H_out, W_out, 2)
        # test_batch_z = np.random.randint(low=0, high=2, size=(batch_size, H_out, W_out, Q))
        # test_loss_val, train_summaries = sess.run(
        #     [loss, test_summaries_tf],
        #     feed_dict={
        #         L_tf: test_batch_x,
        #         z_true_tf: test_batch_z,
        #         pts_in_hull_tf: np.load('pts_in_hull.npy').astype(float)
        #     })
        # print('TEST Step = %04d\tloss = %.4f' % (step + 1,
        #                                          test_loss_val))

        # test_writer.add_summary(train_summaries, step)

print("Run the command line:\n"
      "--> tensorboard --logdir={} "
      "\nThen open http://0.0.0.0:6006/ into your web browser".format(train_dir))
