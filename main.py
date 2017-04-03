'''
Created on 15 Mar 2017

@author: JC Bailey
'''

# Attempt to import libraries
try:
    print "Importing Libraries...",
    import cv2 as cv    # OpenCV    
    import numpy as np  # NumPy
    import glob
    print " SUCCESS."
    
except ImportError, errorMessage:
    print " FAILED. \n\nError was: \"%s\" . Program exiting." % (errorMessage)
    quit() 



def FetchImages ():
    '''
    Fetches all the file names within the Images folder
    '''
    return glob.glob("Images/*.jpg")



if __name__ == '__main__':
    imageNames = FetchImages()
    for fn in imageNames:
        img = cv.imread(fn)
        cv.imshow("Current Image",img)
        cv.waitKey(1)
    print cv.__version__
    
