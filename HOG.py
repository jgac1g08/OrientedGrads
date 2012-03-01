from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy
import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import sys
from optparse import OptionParser

def gradextractver(image):
    kernal = np.array([[0, -1, 0],[0,0,0],[0,1,0]])
    gradImage = signal.convolve(image,kernal)
    return gradImage

def gradextracthor(image):
    kernal = np.array([[0, 0, 0],[1,0,-1],[0,0,0]])
    gradImage = signal.convolve(image,kernal)
    return gradImage

def orientator(imVer,imHor):
    orientation = np.arctan(np.divide(imHor,imVer))
    return orientation

def cellulator(orients,histrange,cellsize=6,histbins=9):
    cells = np.zeros([orients.shape[0]/cellsize,orients.shape[1]/cellsize,histbins])
    xcellrange = orients.shape[0] - cellsize
    ycellrange = orients.shape[1] - cellsize
    for cellx in range(0,xcellrange,cellsize):
        for celly in range(0,ycellrange,cellsize):
            cells[cellx/cellsize,celly/cellsize], _ = np.histogram(orients[cellx:cellx+cellsize,celly:celly+cellsize],histbins,histrange)
            #print np.histogram(orients[cellx:cellx+cellsize,celly:celly+cellsize],histbins,histrange)
    return cells

def normaliser(cells,blocksize=3,e1=0.1): #n.b. edge cells not normalised as it's a faff
    normcells = cells
    for blockx in range(blocksize//2,cells.shape[0]-blocksize//2):
        for blocky in range(blocksize//2,cells.shape[1]-blocksize//2):
            normcells[blockx,blocky] = cells[blockx,blocky] / np.sqrt(np.square(np.linalg.norm(cells[blockx-blocksize//2:blockx+blocksize//2,blocky-blocksize//2:blocky+blocksize//2])) + np.square(e1))
    return normcells

def HOG(img,sign=False,cellsize=6,blocksize=3,histbins=9):
    #code here
    # get the gradients
    gradImgVer  = gradextractver(img)
    gradImgHor  = gradextracthor(img)
    #get the orientations
    orients = orientator(gradImgVer,gradImgHor)
    # signed or unsigned orientations
    if not sign:
        orients = np.absolute(orients)
        histrange = [0,np.pi/2]
    else:
        histrange = [-np.pi/2,np.pi/2]
    # get the cell histograms
    cells = cellulator(orients,histrange,cellsize,histbins)
    # normalise cell histograms over blocks
    normcells = normaliser(cells,blocksize)
    
    return orients, normcells

def run_prog():
    parser = OptionParser()
    parser.add_option("-s", "--signed", action="store_true", default=False)

    (options, args) = parser.parse_args()
    
    imgin = Image.open("images/pedestrian1.png")
    #imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point  
    
    orients, normcells = HOG(img,options.signed)

    plt.set_cmap(plt.cm.gray)
    plt.imshow(orients)
    plt.show()



if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
    
