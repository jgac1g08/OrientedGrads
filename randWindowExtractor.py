from __future__ import division
import Image
import scipy
import numpy as np
import sys
from defaultconfig import default


def randWindowExtractor(img,windowSize=None):
    
    if windowSize == None:
        windowSize = default['window_pixel_shape']
        
    startx = np.random.randint(0,img.shape[0]-windowSize[0])
    starty = np.random.randint(0,img.shape[1]-windowSize[1])

    croppedImg = img[startx:startx+windowSize[0],starty:starty+windowSize[1]]

    return croppedImg
