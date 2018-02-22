# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
TensorBoard was used for vizualization
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.app.flags.DEFINE_string('train_dir', 'runs/',
                           """Output dir for tensorflow summaries.""")
tf.app.flags.DEFINE_string('prefix', 'prefix',
                           """Model name prefix.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-2,
                          """Learning rate.""")
tf.app.flags.DEFINE_integer('n_steps', 1000,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Batch size in SGD.""")

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
input_tf = tf.placeholder(tf.float32, [None, 784], name='input_tf')
with tf.name_scope('fc'):
    x = tf.layers.dense(inputs=input_tf,
                        units=10,
                        name='fc_layer')

# Predicted values
y_pred = tf.nn.softmax(x, name='predicted_labels')
# Ground truth
y_true = tf.placeholder(tf.float32, [None, 10], name='true_labels')

# Metrics
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x,
                                                                  labels=y_true),
                          name='loss')
    loss_summary_tf = tf.summary.scalar("loss", loss)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary_tf = tf.summary.scalar("accuracy", accuracy)

global_step = tf.Variable(0, trainable=False, name='global_step')

# Optimizer
with tf.name_scope("optimizer"):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                           global_step=global_step)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

train_summaries_tf = tf.summary.merge_all()
test_summaries_tf = tf.summary.merge([loss_summary_tf, accuracy_summary_tf])

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
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, train_loss_val, train_accuracy_val, s = sess.run(
            [train_step, loss, accuracy, train_summaries_tf],
            feed_dict={input_tf: batch_xs,
                       y_true: batch_ys})

        print('TRAIN Step = %06d\tloss = %.4f\taccuracy = %.4f' % (step + 1,
                                                                   train_loss_val.mean(),
                                                                   train_accuracy_val))
        train_writer.add_summary(s, step)

        # Cross validate
        test_batch_xs, test_batch_ys = mnist.test.next_batch(batch_size)
        test_loss_val, test_accuracy_val, s = sess.run(
            [loss, accuracy, test_summaries_tf],
            feed_dict={input_tf: batch_xs,
                       y_true: batch_ys})

        print('TEST Step = %06d\tloss = %.4f\taccuracy = %.4f' % (step + 1,
                                                                  test_loss_val.mean(),
                                                                  test_accuracy_val))
        test_writer.add_summary(s, step)


    print('Final accuracy = %.4f' % sess.run(accuracy,
                                             feed_dict={input_tf: mnist.test.images,
                                                        y_true: mnist.test.labels}))

print("Run the command line:\n"
      "--> tensorboard --logdir={} "
      "\nThen open http://0.0.0.0:6006/ into your web browser".format(train_dir))
