# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
# pylint: disable=C0103,W1401
import os
import numpy as np
import sklearn.neighbors as nn
from skimage import io, color
from matplotlib.pyplot import imshow



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


class PriorFactor():
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
        map = dict()
        for ind in range(len(self.prior_factor)):
            map[ind] = pc.prior_factor[ind]
        inv_map = {v: k for k, v in map.items()}
        return np.vectorize(inv_map.get)(prior_Qimage)


def convert_image_Qspace(filepath, NN, sigma, gamma, alpha, ENC_DIR=''):
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
    rgb = io.imread(os.path.join(ENC_DIR, filepath))
    lab = np.array([color.rgb2lab(rgb)]).transpose((0, 3, 1, 2))  # size NxXxYx3

    # Slice the image in L and ab slices
    lab_ab = lab[:, 1:, :, :]

    # km_filepath is a np array with the coordinates of the 313 classes
    nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(ENC_DIR, 'pts_in_hull.npy'))

    N = lab.shape[0]
    X = lab.shape[2]
    Y = lab.shape[3]
    Q = nnenc.K

    encode_lab = nnenc.encode_points_mtx_nd(lab_ab, axis=1)
    encode_lab.reshape(N, Q, X, Y)

    # priorFile np array with class priors computed on the whole ImageNet dataset
    pc = PriorFactor(alpha, gamma=gamma, verbose=False, priorFile=os.path.join(ENC_DIR, 'prior_probs.npy'))
    res = pc.forward(encode_lab, axis=1)
    res.reshape(N, 1, X, Y)

    return res

if __name__ == '__main__':
    NN_ = 10.
    sigma_ = 5.
    gamma_ = .5
    alpha_ = 1.

    filepath_ = 'kitten.jpg'

    prior_Qimage = convert_image_Qspace(filepath_, NN_, sigma_, gamma_, alpha_, ENC_DIR='')
    imshow(prior_Qimage[0, 0])

    # Now retrieve image from Q space to ab space

    pc = PriorFactor(alpha_, gamma=gamma_, verbose=False, priorFile=os.path.join('', 'prior_probs.npy'))
    Qimage = pc.decode(prior_Qimage[0, 0])

    nnenc = NNEncode(NN_, sigma_, km_filepath=os.path.join('', 'pts_in_hull.npy'))

    res = nnenc.decode_without_luminosity(Qimage, L=50)
    imshow(res) # Wrong luminosity, wrong colors but shapes ok

    # Cheat: retrieve luminosity from original image
    rgb = io.imread(os.path.join('', filepath_))
    lab = np.array([color.rgb2lab(rgb)]).transpose((0, 3, 1, 2))  # size NxXxYx3
    L = lab[:, 0, :, :]

    res = nnenc.decode_with_luminosity(Qimage, L[0])
    imshow(res) # Works! but the luminosity is lost
