# -*- coding: utf-8 -*-
"""
Load data helpers
"""
# pylint: disable=invalid-name, redefined-outer-name
import numpy as np


# input: path to the different files
# output: X_train, y_train, X_l

def load_data(a_file, b_file, L_file, gray_file):
    """
    Docstring @PH
    Args:
        a_file ():
        b_file ():
        L_file ():
        gray_file ():

    Returns:

    """
    X_a = np.load(a_file)
    X_b = np.load(b_file)
    X_l = np.load(L_file)
    X_gray = np.load(gray_file)

    X_train = np.zeros((X_gray.shape[0], X_gray.shape[1], 1))
    X_train[:, :, 0] = X_gray

    y_train = np.zeros((X_gray.shape[0], X_gray.shape[1], 2))
    y_train[:, :, 0] = X_a
    y_train[:, :, 1] = X_b

    return X_train, y_train, X_l


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
