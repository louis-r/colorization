# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
from skimage import io, color
import glob
from scipy import ndimage, misc


# input: path to the different files
#output: X_train, y_train, X_l

def load_data(a_file, b_file, L_file, gray_file):
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
    Batch = {}
    Nb_batch = len(X_train) // size_batch + 1
    for i in range(Nb_batch - 1):
        Batch[i] = (X_train[i * size_batch:(i + 1) * size_batch], y_train[i * size_batch:(i + 1) * size_batch])
    Batch[Nb_batch] = (X[i * (Nb_batch - 1):], y_train[i * (Nb_batch - 1):])

    return Batch

