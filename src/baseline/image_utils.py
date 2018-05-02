# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
# pylint: disable=C0103, W1401, redefined-outer-name
import os
import numpy as np
from skimage import io, color
from matplotlib.pyplot import imshow
import sklearn.neighbors as nn
import tensorflow as tf


def check_value(inds, val):
    """
    Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function
    """
    if np.array(inds).size == 1:
        if inds == val:
            return True
    return False


def na():  # shorthand for new axis
    return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
    """
    Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array
    """
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array(axis))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    pts_flt = pts_nd.transpose(axorder)
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    """
    Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array
        """
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array(axis))  # non axis indices
    # NPTS = np.prod(SHP[nax])

    if squeeze:
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out


class NNEncode:
    """
    Encode points using NN search and Gaussian kernel
    """

    def __init__(self, NN, sigma, km_filepath='', cc=-1):
        if check_value(cc, -1):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        # Fixed code here
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, sameBlock=True):
        """
        Missing docstring
        Args:
            pts_nd ():
            axis ():
            returnSparse ():
            sameBlock ():

        Returns:

        """
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]
        if sameBlock and self.alreadyUsed:
            self.pts_enc_flt[...] = 0  # already pre-allocated # pylint: disable=access-member-before-definition
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        P = pts_flt.shape[0]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]

        self.pts_enc_flt[self.p_inds, inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt, self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self, pts_enc_nd, axis=1, returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd, axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd, axis=axis)
        if returnEncode:
            return pts_dec_nd, pts_1hot_nd
        return pts_dec_nd

    def decode_with_luminosity(self, Qimage, L):
        # Converts image in Q space to image in ab space with default luminosity
        res = np.array([L, self.cc[Qimage, 0], self.cc[Qimage, 1]]).transpose(1, 2, 0)
        return color.lab2rgb(res)

    def decode_without_luminosity(self, Qimage, L=50):
        # Converts image in Q space to image in ab space with default luminosity
        res = np.array([L * np.ones(Qimage.shape), self.cc[Qimage, 0], self.cc[Qimage, 1]]).transpose(1, 2, 0)
        return color.lab2rgb(res)


class PriorFactor:
    """
    Class handles prior factor
    """

    def __init__(self, alpha, gamma=0, verbose=True, priorFile=''):
        """

        Args:
            alpha (): prior correction factor, 0 to ignore prior, 1 to divide by prior,
                        alpha to divide by prior**alpha
            gamma (): percentage to mix in uniform prior with empirical prior
            verbose ():
            priorFile (): file which contains prior probabilities across classes
        """
        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix ** -self.alpha
        self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs * self.prior_factor
        self.implied_prior = self.implied_prior / np.sum(self.implied_prior)  # re-normalize

        if self.verbose:
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' % (
            np.min(self.prior_factor), np.max(self.prior_factor), np.mean(self.prior_factor),
            np.median(self.prior_factor),
            np.sum(self.prior_factor * self.prior_probs)))

    def forward(self, data_ab_quant, axis=1):  # pylint: disable=inconsistent-return-statements
        """
        Missing docstring
        Args:
            data_ab_quant ():
            axis ():

        Returns:

        """
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if axis == 0:
            return corr_factor[na(), :]
        elif axis == 1:
            return corr_factor[:, na(), :]
        elif axis == 2:
            return corr_factor[:, :, na(), :]
        elif axis == 3:
            return corr_factor[:, :, :, na()]

    def decode(self, prior_Qimage):
        map_dict = dict()
        for ind in range(len(self.prior_factor)):
            map_dict[ind] = pc.prior_factor[ind]
        inv_map = {v: k for k, v in map_dict.items()}
        return np.vectorize(inv_map.get)(prior_Qimage)


def convert_image_Qspace(lab_ab, NN, sigma, gamma, alpha, ENC_DIR=''):
    """
    Missing docstring
    Args:
        filepath ():
        NN ():
        sigma ():
        gamma ():
        alpha ():
        ENC_DIR ():

    Returns:

    """
    # rgb = io.imread(os.path.join(ENC_DIR, filepath))
    # lab = np.array([color.rgb2lab(rgb)]).transpose((0, 3, 1, 2))  # size NxHxWx3
    # lab shape N, 3, H, W

    # Slice the image in L and ab slices
    # lab_ab = lab[:, 1:, :, :]

    # km_filepath is a np array with the coordinates of the 313 classes
    nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(ENC_DIR, 'pts_in_hull.npy'))

    N = lab_ab.shape[0]
    H = lab_ab.shape[2]
    W = lab_ab.shape[3]
    Q = nnenc.K

    encode_lab = nnenc.encode_points_mtx_nd(lab_ab, axis=1)
    encode_lab.reshape(N, Q, H, W)

    # priorFile np array with class priors computed on the whole ImageNet dataset
    pc = PriorFactor(alpha,
                     gamma=gamma,
                     verbose=False,
                     priorFile=os.path.join(ENC_DIR, 'prior_probs.npy'))

    res = pc.forward(encode_lab, axis=1)
    res.reshape(N, 1, H, W)

    return res, encode_lab


# Tensorflow
def check_image(image):
    """
    Checks tensorflow imag dimensions
    Args:
        image (tf.Tensor):

    Returns:
        Image with last dimension 3
    """
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def lab_to_rgb(lab):
    """
    Converts tensorflow LAB image to RGB
    Args:
        lab (tf.Tensor):

    Returns:

    """
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + \
                         (fxfyfz_pixels ** 3) * exponential_mask

            # Denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (
                1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


if __name__ == '__main__':
    NN_ = 10.
    sigma_ = 5.
    gamma_ = .5
    alpha_ = 1.

    filepath_ = 'kitten.jpg'

    prior_Qimage = convert_image_Qspace(filepath_, NN_, sigma_, gamma_, alpha_, ENC_DIR='')
    imshow(prior_Qimage[0, 0])  # pylint: disable=invalid-sequence-index

    # Now retrieve image from Q space to ab space
    pc = PriorFactor(alpha_, gamma=gamma_, verbose=False, priorFile=os.path.join('', 'prior_probs.npy'))
    Qimage = pc.decode(prior_Qimage[0, 0])  # pylint: disable=invalid-sequence-index

    nnenc = NNEncode(NN_, sigma_, km_filepath=os.path.join('', 'pts_in_hull.npy'))

    res = nnenc.decode_without_luminosity(Qimage, L=50)
    imshow(res)  # Wrong luminosity, wrong colors but shapes ok

    # Cheat: retrieve luminosity from original image
    rgb = io.imread(os.path.join('', filepath_))
    lab = np.array([color.rgb2lab(rgb)]).transpose((0, 3, 1, 2))  # size NxXxYx3
    L = lab[:, 0, :, :]

    res = nnenc.decode_with_luminosity(Qimage, L[0])
    imshow(res)  # Works! but the luminosity is lost
