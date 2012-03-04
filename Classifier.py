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

window_pixel_shape = default['window_pixel_shape']

window_move_stride = default['window_move_stride']

cellsize = 6

def run_prog():
    parser = OptionParser()
    parser.add_option("-m", "--model-file", action="store", default="svm_model.pkl")
    parser.add_option("-s", "--signed", action="store_true", default=False)
    parser.add_option("-r", "--random-window", action="store_true", default=False)

    (options, args) = parser.parse_args()
    
    print "Loading model from file ", options.model_file
    model_file = open(options.model_file, "rb")
    scaler, svc = pickle.load(model_file)
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
    
	#######################################################################
    #######################################################################
    #######################################################################
    #### http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use ####
    #######################################################################
    #######################################################################
    #######################################################################
    
    #orients, normcells = HOG.HOG(img, options.signed)
    
    #//if normcells.shape[0] <= window_shape[0]:
    #//    cellx_max = 1
    #//else:
    #//    cellx_max = normcells.shape[0] - window_shape[0]
    #//    
    #//if normcells.shape[1] <= window_shape[1]:
    #//    celly_max = 1
    #//else:
    #//    celly_max = normcells.shape[1] - window_shape[1]
    
    
    
    
    if options.random_window:
        img = randWindowExtractor(img)
    
    print "Image shape", img.shape
    
    
    window_hits = np.zeros((img.shape[0] // window_move_stride, img.shape[1] // window_move_stride))
    
    if False:
        for x in range(0, img.shape[0], window_move_stride):
            print x // window_move_stride, "of", window_hits.shape[0]

            if (x + window_pixel_shape[0]) > img.shape[0]:
                break
            for y in range(0, img.shape[1], window_move_stride):
                if (y + window_pixel_shape[1]) > img.shape[1]:
                    break
                    
                window = img[x:x + window_pixel_shape[0], y:y + window_pixel_shape[1]]
                orients, normcells = HOG.HOG(window, options.signed)
                normcells = normcells.flatten()
                normcells = scaler.transform(normcells)
                prediction = svc.predict(normcells)
                #print "Prediction at", x // window_move_stride, y // window_move_stride, "of", window_hits.shape, "is", prediction
                window_hits[x // window_move_stride, y // window_move_stride] = prediction

    if True:
        window_hits = np.zeros((img.shape[0] // cellsize, img.shape[1] // cellsize))
        
        window_cell_shape = (window_pixel_shape[0] // cellsize, window_pixel_shape[1] // cellsize)
        
        print "Image cells", window_hits.shape
        print "Window cells", window_cell_shape
        
        orients, normcells = HOG.HOG(img, options.signed)
        
        print "img shape", img.shape
        print "normcells shape", normcells.shape
        print "orients shape", orients.shape
        
        
        print "Actual image cells", normcells.shape
        
        for x in range(0, window_hits.shape[0]):
            print x, "of", window_hits.shape[0]

            if (x + window_cell_shape[0]) > window_hits.shape[0]:
                break
            for y in range(0, window_hits.shape[1]):
                if (y + window_cell_shape[1]) > window_hits.shape[1]:
                    break
                
                window = normcells[x:x + window_cell_shape[0], y:y + window_cell_shape[1]]
                window = window.flatten()
                window = scaler.transform(window)
                prediction = svc.predict(window)
                
                if prediction >= 0.5:
                    print "prediction", prediction
                #print "Prediction at", x // window_move_stride, y // window_move_stride, "of", window_hits.shape, "is", prediction
                window_hits[x, y] = prediction
                #window_hits[x , y] = 1
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.set_cmap(plt.cm.gray)
    ax.imshow(img)
    #ax.add_patch(matplotlib.patches.Rectangle(((window_hits[0] * cellsize) + window_pixel_shape[0], (window_hits[1] * cellsize) + window_pixel_shape[1]), window_pixel_shape[0], window_pixel_shape[1]))
    for x_win_hits in range(window_hits.shape[0]):
        for y_win_hits in range(window_hits.shape[1]):
            #ax.add_patch(matplotlib.patches.Rectangle((y_win_hits * cellsize, x_win_hits * cellsize), window_pixel_shape[1], window_pixel_shape[0], ec='blue', facecolor='none'))
            if window_hits[x_win_hits, y_win_hits] >= 0.5:
                print (x_win_hits * cellsize), (y_win_hits * cellsize)
                ax.add_patch(matplotlib.patches.Rectangle((y_win_hits * cellsize, x_win_hits * cellsize), window_pixel_shape[1], window_pixel_shape[0], ec='red', facecolor='none', hatch="/"))
                
    plt.show()

if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
