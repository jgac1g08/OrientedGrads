import os
import Image
import sys

class DirectoryImagesLoader:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.dirlist = os.listdir(self.dirpath)
        self.dirlist.sort()        
    
    def get_image(self, number, to_grey=True, to_float=False):
        imgin = Image.open(os.path.join(dirpath, dirlist[number]))

        if to_grey:
            imgin = imgin.convert("L") # convert to greyscale (luminance)
    
        if to_float:
            img = np.asarray(imgin)
            img = img.astype(np.float32) # convert to a floating point
            
        return img
    
    def __len__(self):
        return len(self.dirlist)
        
    
if __name__ == "__main__":
    imgs = DirectoryImagesLoader(sys.argv[1])
    print "Images:", imgs.dirlist
    print "Number of images:", len(imgs)
