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
from sklearn import svm
import cPickle as pickle

window_shape = [22, 12]

window_move_step = 1

def run_prog():
    parser = OptionParser()
    parser.add_option("-m", "--model-file", action="store", default="svm_model.pkl")
    parser.add_option("-s", "--signed", action="store_true", default=False)

    (options, args) = parser.parse_args()
    
    print "Loading model from file ", options.model_file
    model_file = open(options.model_file, "rb")
    svc = pickle.load(model_file)
    model_file.close()
    print "Model loaded"
    
    #imgin = Image.open("images/pedestrian1.png")
    imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    #img = img.astype(np.float32) # convert to a floating point  
    
    #######################################################################
    #######################################################################
    #######################################################################
    #### try doing windowing before we do the HOG and just extract the ####
    #### 					HOG for each window						   ####
    #######################################################################
    #######################################################################
    #######################################################################
    
    orients, normcells = HOG.HOG(img, options.signed)
    
    if normcells.shape[0] <= window_shape[0]:
        cellx_max = 1
    else:
        cellx_max = normcells.shape[0] - window_shape[0]
        
    if normcells.shape[1] <= window_shape[1]:
        celly_max = 1
    else:
        celly_max = normcells.shape[1] - window_shape[1]
    
    
    window_hits = np.zeros((cellx_max, celly_max))
    
    
    
    for cellx in range(0, cellx_max, window_move_step):
        for celly in range(0, celly_max, window_move_step):
            print normcells[cellx:cellx + window_shape[0], celly:celly + window_shape[1]].shape
            prediction = svc.predict(normcells[cellx:cellx + window_shape[0], celly:celly + window_shape[1]].flatten())
            print "Prediction at", cellx, celly, "is", prediction
            window_hits[cellx, celly] = prediction
    
    
    
    
    plt.set_cmap(plt.cm.gray)
    plt.imshow(window_hits)
    plt.show()

if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
