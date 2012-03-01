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
import sPickle
from randWindowExtractor import randWindowExtractor
import cPickle as pickle
from sklearn import svm

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", "--pos-hogs", action="store", default="training_hogs_pos.pkl")
    parser.add_option("-n", "--neg-hogs", action="store", default="training_hogs_neg.pkl")
    parser.add_option("-m", "--model-save", action="store", default="svm_model.pkl")
    
    (options, args) = parser.parse_args()    
    
    print "Loading training data from file"
    pos_hogs_file = open(options.pos_hogs, 'rb')
    neg_hogs_file = open(options.neg_hogs, 'rb')
    
    pos_hogs = np.asarray(pickle.load(pos_hogs_file))
    neg_hogs = np.asarray(pickle.load(neg_hogs_file))

    pos_hogs_file.close()
    neg_hogs_file.close()
    
    print "Data loaded, sir!"
    
    training_data = np.concatenate([pos_hogs, neg_hogs])
    
    
    training_labels = np.append([0] * len(pos_hogs), [1] * len(neg_hogs))
    
    svc = svm.SVC(C=0.01)
    svc.fit(training_data, training_labels)
    
    print "Should be pos (0), result:", svc.predict(pos_hogs[0])
    print "Should be neg (1), result:", svc.predict(neg_hogs[0])
    
    
    print "Saving model to", options.model_save
    
    model_file = open(options.model_save, "wb")
    pickle.dump(svc, model_file)
    model_file.close()
    
    print "Model saved"
    
    print "Finished!"
        
        
