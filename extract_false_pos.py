from __future__ import division
import matplotlib.pyplot as plt
import matplotlib
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
from defaultconfig import default
from randWindowExtractor import randWindowExtractor
from Classifier import detect_humans
from imageloader import DirectoryImagesLoader

def run_prog():
    parser = OptionParser()
    parser.add_option("-m", "--model-file", action="store", default="svm_model.pkl")
    parser.add_option("-f", "--false-pos-file", action="store", default="false_pos.pkl")
    parser.add_option("-n", "--num-images", action="store", type=int, default=-1)
    parser.add_option("-s", "--signed", action="store_true", default=False)
    parser.add_option("-r", "--random-window", action="store_true", default=False)

    (options, args) = parser.parse_args()
    
    print "Loading model from file ", options.model_file
    model_file = open(options.model_file, "rb")
    scaler, svc = pickle.load(model_file)
    model_file.close()
    print "Model loaded"
    
    
    imgs = DirectoryImagesLoader(args[0])
    imgs.randomize()
    
    false_pos_cells = []
    
    if options.num_images != -1:
        stop_at = options.num_images
    else:
        stop_at = len(imgs)
    
    for i in range(stop_at):
        print "Extracting false positives on image", i, "of", stop_at
        img = imgs.get_image(i)
        window_hits, detected_humans = detect_humans(img, svc, scaler, options.signed, debug=False, extract_humans_to=false_pos_cells)
        print len(detected_humans), "humans found"
    
    #ax.add_patch(matplotlib.patches.Rectangle((y_win_hits * cellsize, x_win_hits * cellsize), window_pixel_shape[1], window_pixel_shape[0], ec='red', facecolor='none', hatch="/"))
    
    f = open(options.false_pos_file, 'wb')
    
    pickle.dump(false_pos_cells, f)
    
    f.close()
            
if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
