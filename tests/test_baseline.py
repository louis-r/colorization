import pytest
import tensorflow as tf

import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../src/baseline"))
from baseline_model import *


@pytest.fixture
def x():
    out = tf.constant([[[[1, 1, 1],
                         [2, 2, 2]],
                        [[3, 3, 3],
                         [4, 4, 4]]]])
    return out


@pytest.fixture
def pad1():
    expected_padded_x = tf.constant([[[[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [1, 1, 1],
                                       [2, 2, 2],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [3, 3, 3],
                                       [4, 4, 4],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]]]])
    return expected_padded_x


@pytest.fixture
def pad2():
    expected_padded_x = tf.constant([[[[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [1, 1, 1],
                                       [2, 2, 2],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [3, 3, 3],
                                       [4, 4, 4],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]],
                                      [[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]]]])
    return expected_padded_x


@pytest.fixture
def sess():
    return tf.Session()


def test_return_padded(x, pad1, pad2, sess):
    pad = 0
    padded = sess.run(return_padded(x, pad))
    var = sess.run(tf.reduce_min(tf.cast(tf.equal(padded, x), tf.float32)))
    assert var

    pad = 1
    padded = sess.run(return_padded(x, pad))
    equality = tf.equal(padded, pad1)
    var = sess.run(tf.reduce_min(tf.cast(equality, tf.float32)))
    assert var

    pad = 2
    padded = sess.run(return_padded(x, pad))
    equality = tf.equal(padded, pad2)
    var = sess.run(tf.reduce_min(tf.cast(equality, tf.float32)))
    assert var
