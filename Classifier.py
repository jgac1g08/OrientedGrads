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



cellsize = 6

window_cell_shape = default['window_cell_shape']

window_pixel_shape = (window_cell_shape[0] * cellsize, window_cell_shape[1] * cellsize)

def detect_humans(img, svc, scaler, signed=False, debug=True, extract_humans_to=None):
    window_hits = np.zeros([img.shape[0] // cellsize, img.shape[1] // cellsize])
    detected_humans = []
    
    window_cell_shape = (window_pixel_shape[0] // cellsize, window_pixel_shape[1] // cellsize)
    
    
    orients, normcells = HOG.HOG(img, signed, cellsize=cellsize)
    
    #detected_humans.append((0, 0, 1.0))
    
    for x in range(0, img.shape[0] // cellsize):
        if debug:
            print x, "of", img.shape[0] // cellsize

        if (x + window_cell_shape[0]) > img.shape[0] // cellsize:
            break
        for y in range(0, img.shape[1] // cellsize):
            if (y + window_cell_shape[1]) > img.shape[1] // cellsize:
                break
            
            window = normcells[x:x + window_cell_shape[0], y:y + window_cell_shape[1]]
            window = window.flatten()
            window_s = scaler.transform(window)
            prediction = svc.predict(window_s)               
            
            
            if prediction >= 0.5:
                if debug:
                    print "Prediction", prediction
                detected_humans.append((x * cellsize, y * cellsize, prediction))
                if extract_humans_to is not None:
                    extract_humans_to.append(window)
            #print "Prediction at", x // window_move_stride, y // window_move_stride, "of", window_hits.shape, "is", prediction
            window_hits[x, y] = prediction
            
    
    return window_hits, detected_humans

def run_prog():
    parser = OptionParser()
    parser.add_option("-m", "--model-file", action="store", default="svm_model.pkl")
    parser.add_option("-s", "--signed", action="store_true", default=False)
    parser.add_option("-r", "--random-window", action="store_true", default=False)
    parser.add_option("-n", "--num-scales", action="store", type=int, default=1)

    (options, args) = parser.parse_args()
    
    print "Loading model from file ", options.model_file
    model_file = open(options.model_file, "rb")
    scaler, svc = pickle.load(model_file)
    model_file.close()
    print "Model loaded"
    
    #imgin = Image.open("images/pedestrian1.png")
    imgin = Image.open(args[0])
    
    #imgin = imgin.resize((imgin.size[0] // 2, imgin.size[1] // 2), Image.CUBIC)
	
    humans_scale = []
    
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    if options.num_scales == 1:
        scales = [1]
    else:
        scales = np.linspace(1, 0.2, num=5)
    
    for i in range(len(scales)):
        scale = scales[i]
        print "Doing scale ", scale, i, "of", len(scales)
        
        img = imgin.resize((int(imgin.size[0] * scale), int(imgin.size[1] * scale)), Image.LINEAR)
        
        img = np.array(img, dtype='d', order='C')
        
        if options.random_window:
            img = randWindowExtractor(img, options.signed)
        
        print "Image shape", img.shape
        

        window_hits, detected_humans = detect_humans(img, svc, scaler, options.signed, debug=True)
        humans_scale.append((scale, detected_humans))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.set_cmap(plt.cm.gray)
    ax.imshow(np.array(imgin))
    #ax.add_patch(matplotlib.patches.Rectangle(((window_hits[0] * cellsize) + window_pixel_shape[0], (window_hits[1] * cellsize) + window_pixel_shape[1]), window_pixel_shape[0], window_pixel_shape[1]))
    #for x_win_hits in range(window_hits.shape[0]):
    #    for y_win_hits in range(window_hits.shape[1]):
    #        #ax.add_patch(matplotlib.patches.Rectangle((y_win_hits * cellsize, x_win_hits * cellsize), window_pixel_shape[1], window_pixel_shape[0], ec='blue', facecolor='none'))
    #        if window_hits[x_win_hits, y_win_hits] >= 0.5:
    #            print (x_win_hits * cellsize), (y_win_hits * cellsize)
    #            ax.add_patch(matplotlib.patches.Rectangle((y_win_hits * cellsize, x_win_hits * cellsize), window_pixel_shape[1], window_pixel_shape[0], ec='red', facecolor='none', hatch="/"))
    
    for scale, detected_humans in humans_scale:
        for x, y, _ in detected_humans:
            ax.add_patch(matplotlib.patches.Rectangle((y, x), window_pixel_shape[1] / scale, window_pixel_shape[0] / scale, ec='red', facecolor='none', hatch="/"))
    
    plt.figure()
    plt.imshow(window_hits)
    plt.show()

if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
