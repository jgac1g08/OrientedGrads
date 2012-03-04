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
import sklearn

import random

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
    
    # randomly segment into training and test set
    
    #test_proportion = 0.1
    test_size = 300
    
    pos_rand = random.sample(pos_hogs, len(pos_hogs)) # choose k unique values
    neg_rand = random.sample(neg_hogs, len(neg_hogs))
    
    # get training data but leave test_size samples for testing data
    training_pos = pos_rand[:-test_size]
    training_neg = neg_rand[:-test_size]
    
    testing_pos = pos_rand[-test_size:]
    testing_neg = neg_rand[-test_size:]
    
    training_data = np.concatenate([training_pos, training_neg])
    
    scaler = sklearn.preprocessing.Scaler().fit(training_data)
    training_data = scaler.transform(training_data)
    
    training_labels = np.append([1] * len(training_pos), [0] * len(training_neg))
    
    svc = svm.SVC(C=0.1, kernel='linear', probability=True, scale_C=True)
    svc.fit(training_data, training_labels)
    
    testing_data =  np.concatenate([testing_pos, testing_neg])
    
    testing_data = scaler.transform(testing_data)
    
    testing_labels = np.append([1] * len(testing_pos), [0] * len(testing_neg))
 
    #for i in range(len(testing_data)):
		#print "Should be", testing_labels[i] ,", result:", svc.predict(testing_data[i])
        
    print "Test score:", svc.score(testing_data, testing_labels) 
    
    test_probas = svc.predict_proba(testing_data)
    
    # Compute ROC curve and area the curve - code from http://scikit-learn.org/0.10/auto_examples/plot_roc.html
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(testing_labels, test_probas[:, 1])
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
    
    print "Saving model to", options.model_save
    
    model_file = open(options.model_save, "wb")
    pickle.dump((scaler, svc), model_file)
    model_file.close()
    
    print "Model saved"
    
    print "Finished!"
        
        
