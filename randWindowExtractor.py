from __future__ import division
import Image
import scipy
import numpy as np
import sys


def randWindowExtractor(img,windowSize = [70,134]):
    
    startx = np.random.randint(0,img.shape[0]-windowSize[0])
    starty = np.random.randint(0,img.shape[1]-windowSize[1])

    croppedImg = img[startx:startx+windowSize[0],starty:starty+windowSize[1]]

    return croppedImg
