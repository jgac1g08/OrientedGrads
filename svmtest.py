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
from sklearn import svm
import random



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", "--pos-hogs", action="store", default="training_hogs_pos.pkl")
    parser.add_option("-n", "--neg-hogs", action="store", default="training_hogs_neg.pkl")
    parser.add_option("-m", "--model-save", action="store", default="svm_model.pkl")
    
    (options, args) = parser.parse_args()    
    
    print "Loading testing data from file"
    pos_hogs_file = open(options.pos_hogs, 'rb')
    neg_hogs_file = open(options.neg_hogs, 'rb')
    
    pos_hogs = np.asarray(pickle.load(pos_hogs_file))
    neg_hogs = np.asarray(pickle.load(neg_hogs_file))

    pos_hogs_file.close()
    neg_hogs_file.close()
    
    print "Data loaded, sir!"
    
    
    print "Loading model from", options.model_save
    
    model_file = open(options.model_save, "rb")
    svc = pickle.load(model_file)
    model_file.close()
    
    print "Model loaded"
    
    print "Testing..."
    
    testing_pos = random.sample(pos_hogs, 200)
    testing_neg = random.sample(pos_hogs, 200)
    
    testing_data = np.concatenate([testing_pos, testing_neg])
    testing_labels = np.append([1] * len(testing_pos), [0] * len(testing_neg))
    
    
    print "Testing complete! Score:", svc.score(testing_data, testing_labels)
    
    

    
    print "Finished!"
        
        
