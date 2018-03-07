# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
from skimage import io, color

rgb = io.imread('kitten.jpg')
lab = color.rgb2lab(rgb)





