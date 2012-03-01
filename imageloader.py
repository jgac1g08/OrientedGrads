import os
import Image
import sys

import numpy as np

class DirectoryImagesLoader:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.dirlist = os.listdir(self.dirpath)
        self.dirlist.sort()        
    
    def get_image(self, number, to_grey=True, to_float=False):
        imgin = Image.open(os.path.join(self.dirpath, self.dirlist[number]))

        if to_grey:
            imgin = imgin.convert("L") # convert to greyscale (luminance)
    
        imgin = np.asarray(imgin)
        
        if to_float:
            imgin = img.astype(np.float32) # convert to a floating point
            
        return imgin
    
    def __len__(self):
        return len(self.dirlist)
        
    
if __name__ == "__main__":
    imgs = DirectoryImagesLoader(sys.argv[1])
    print "Images:", imgs.dirlist
    for i in range(len(imgs)):
        print imgs.get_image(i)
    print "Number of images:", len(imgs)
