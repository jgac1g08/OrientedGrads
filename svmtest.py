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
import random
import sklearn


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
    scaler, svc = pickle.load(model_file)
    model_file.close()
    
    print "Model loaded"
    
    print "Testing..."
    
    test_size = 200
    
    testing_pos = random.sample(pos_hogs, test_size)
    testing_neg = random.sample(neg_hogs, test_size)
    
    testing_data = np.concatenate([testing_pos, testing_neg])
    
    testing_data = scaler.transform(testing_data)
    
    testing_labels = np.append([1] * len(testing_pos), [0] * len(testing_neg))
    
    
    print "Testing complete! Score:", svc.score(testing_data, testing_labels)

    
    testing_labels = np.append([1] * len(testing_pos), [0] * len(testing_neg))
 
    #for i in range(len(testing_data)):
		#print "Should be", testing_labels[i] ,", result:", svc.predict(testing_data[i])
        
    print "Test score:", svc.score(testing_data, testing_labels) 
    
    test_probas = svc.predict_proba(testing_data)
    
    
    # Compute ROC curve and area the curve - code from http://scikit-learn.org/0.10/auto_examples/plot_roc.html
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(testing_labels, test_probas[:,1])
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    
    # Plot ROC curve - code from http://scikit-learn.org/0.10/auto_examples/plot_roc.html
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    
    print "Finished!"
        
        
