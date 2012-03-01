from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy
import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import sys
import HOG
from optparse import OptionParser


def run_prog():
    parser = OptionParser()
    parser.add_option("-s", "--signed", action="store_true", default=False)

    (options, args) = parser.parse_args()
    
    imgin = Image.open("images/pedestrian1.png")
    #imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point  
    
    orients, normcells = HOG.HOG(img,options.signed)

    plt.set_cmap(plt.cm.gray)
    plt.imshow(orients)
    plt.show()



if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
