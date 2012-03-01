from __future__ import division

import HOG
from imageloader import DirectoryImagesLoader

import matplotlib.pyplot as plt
import Image
import scipy
import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import sys
from optparse import OptionParser

from randWindowExtractor import randWindowExtractor

import cPickle as pickle

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--signed", action="store_true", default=False)
    parser.add_option("-r", "--random-window", action="store_true", default=False)
    
    (options, args) = parser.parse_args()    
    
    if len(args) < 2:
        raise "Must provide image directory and file to save to"
    
    imgs = DirectoryImagesLoader(args[0])
    
    hogs = []
    
    
    
    for i in range(len(imgs)):
        print "Running HOG on image", i, "of", len(imgs)
        img = imgs.get_image(i)
        
        if options.random_window:
            img = randWindowExtractor(img)
            
        _, normcells = HOG.HOG(img, options.signed)
        hogs.append(normcells.flatten())
        #sPickle.s_dump_elt(normcells, f)
        
    f = open(args[1], 'wb')
    
    pickle.dump(hogs, f)
    
    f.close()
    
    print "Finished"
        
        
