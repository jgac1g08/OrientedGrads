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

import multiprocessing


cellsize = 6

window_cell_shape = default['window_cell_shape']

window_pixel_shape = (window_cell_shape[0] * cellsize, window_cell_shape[1] * cellsize)


# some code from http://broadcast.oreilly.com/2009/04/pymotw-multiprocessing-part-2.html
class WindowWorker(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue, img, svc, scaler, normcells):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        
        
        self.img = img
        self.svc = svc
        self.scaler = scaler
        self.normcells = normcells

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            #print '%s: %s' % (proc_name, next_task)
            answer = self.process_window(next_task[0], next_task[1])
            if answer is not None:
                self.result_queue.put(answer)
                print answer
            self.task_queue.task_done()
        return
        
    def process_window(self, x, y):
        window = self.normcells[x:x + window_cell_shape[0], y:y + window_cell_shape[1]]
        window = window.flatten()
        window_s = self.scaler.transform(window)
        prediction = self.svc.predict(window_s)               
    
        if prediction >= 0.5:
            return (x * cellsize, y * cellsize, prediction)
        else:
            return None

def detect_humans(img, svc, scaler, signed=False, debug=True, extract_humans_to=None):
    window_hits = np.zeros([img.shape[0] // cellsize, img.shape[1] // cellsize])
    detected_humans = []
    
    window_cell_shape = (window_pixel_shape[0] // cellsize, window_pixel_shape[1] // cellsize)
    
    
    orients, normcells = HOG.HOG(img, signed, cellsize=cellsize)
    
    #detected_humans.append((0, 0, 1.0))
    
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)
    
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

def detect_humans_multi(img, svc, scaler, signed=False, debug=True, extract_humans_to=None):
    window_hits = np.zeros([img.shape[0] // cellsize, img.shape[1] // cellsize])
    detected_humans = []
    
    window_cell_shape = (window_pixel_shape[0] // cellsize, window_pixel_shape[1] // cellsize)
    
    
    orients, normcells = HOG.HOG(img, signed, cellsize=cellsize)
    
    #detected_humans.append((0, 0, 1.0))
    
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)
    
    window_locations = []
    
    for x in range(0, img.shape[0] // cellsize):
        if debug:
            print x, "of", img.shape[0] // cellsize

        if (x + window_cell_shape[0]) > img.shape[0] // cellsize:
            break
        for y in range(0, img.shape[1] // cellsize):
            if (y + window_cell_shape[1]) > img.shape[1] // cellsize:
                break
            
            window_locations.append((x, y))
    
    # some code from http://broadcast.oreilly.com/2009/04/pymotw-multiprocessing-part-2.html
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    
    num_consumers = multiprocessing.cpu_count() * 2
    print 'Creating %d consumers' % num_consumers
    consumers = [ WindowWorker(tasks, results, img, svc, scaler, normcells) for i in xrange(num_consumers) ]
    
    for w in consumers:
        w.start()

    for window_loc in window_locations:
        tasks.put(window_loc)
    
    # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)
    
    print "Joining"
    tasks.join()
    
    detected_humans = []
    while not results.empty():
        detected_humans.append(results.get())
    
    print "Num results found:", len(detected_humans)
    
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
        

        window_hits, detected_humans = detect_humans_multi(img, svc, scaler, options.signed, debug=True)
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
