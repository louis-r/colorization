# -*- coding: utf-8 -*-
"""
Load data helpers
"""
# pylint: disable=invalid-name, redefined-outer-name
import os
import sys
import numpy as np
# noinspection PyUnresolvedReferences
from image_utils import convert_image_Qspace


def load_data(a_file, b_file, L_file, **kwargs):
    """
    Docstring @PH
    Args:
        a_file ():
        b_file ():
        L_file ():
        gray_file ():

    Returns:

    """

    NN = kwargs.pop('NN_ ', 10.)
    sigma = kwargs.pop('sigma_ ', 5.)
    gamma = kwargs.pop('gamma_ ', .5)
    alpha = kwargs.pop('alpha_ ', 1.)

    X_a = np.load(a_file)
    X_b = np.load(b_file)
    X_l = np.load(L_file)
    # X_gray = np.load(gray_file)

    n_images = 1
    print('Subsetting to the first {} images'.format(n_images))
    X_a = X_a[:n_images, :]
    X_b = X_b[:n_images, :]
    X_l = X_l[:n_images, :]

    lab_ab = np.zeros((X_l.shape[0], X_l.shape[1], 2))
    lab_ab[:, :, 0] = X_a
    lab_ab[:, :, 1] = X_b

    # Reshape
    lab_ab = lab_ab.reshape(-1, 256, 256, 2)
    lab_ab = lab_ab.transpose((0, 3, 1, 2))  # N, 3, H, W
    X_l = X_l.reshape(-1, 256, 256, 1)

    prior_Qimage, Q_image = convert_image_Qspace(lab_ab=lab_ab, NN=NN, sigma=sigma, gamma=gamma, alpha=alpha,
                                                 ENC_DIR='')

    return X_l, Q_image


# creates a dictionary of batches of data
def dict_batch(X_train, y_train, size_batch):
    """
    Docstring @PH
    Args:
        X_train ():
        y_train ():
        size_batch ():

    Returns:

    """
    Batch = {}
    Nb_batch = len(X_train) // size_batch + 1
    for i in range(Nb_batch - 1):
        Batch[i] = (X_train[i * size_batch:(i + 1) * size_batch], y_train[i * size_batch:(i + 1) * size_batch])
    Batch[Nb_batch] = (X[size_batch * (Nb_batch - 1):], y_train[size_batch * (Nb_batch - 1):])

    return Batch
