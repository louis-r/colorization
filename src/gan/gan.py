import os
import sys
import time
import math
import cPickle
from scipy import misc
from glob import glob

from ops_new import *
from utils_new import *


class WGAN(object):
    def __init__(self, sess, config=None):
        self.sess = sess
        self.config = config
        self.build_model(config)

    def build_model(self, config):
        pass

    def train(self, config=None):
        pass
    
    def discriminator(self, image, y=None, reuse=False, config=None):
        pass

    def generator(self, z, image_Y, config=None):
        pass
    
    def save(self, config=None, step=0):
        pass

    def load(self, config=None):
        pass

