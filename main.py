'''
Created on 15 Mar 2017

@author: JC Bailey
'''

# Attempt to import libraries
try:
    print "Importing Libraries...",
    import cv2 as cv    # OpenCV    
    import numpy as np  # NumPy
    from OpenCVExtended import *
    import glob
    print " SUCCESS."
    
except ImportError, errorMessage:
    print " FAILED. \n\nError was: \"%s\" . Program exiting." % (errorMessage)
    quit() 



def FetchImages ():
    '''
    Fetches all the file names within the Images folder
    '''
    fns = glob.glob("Images/*.jpg")
    imgs = [cv.imread(fn) for fn in fns]
    return imgs


if __name__ == '__main__':
    image = FetchImages()[0]
    bw_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    g_img = cv.GaussianBlur(bw_image,(3,3),1.0)
    at_img = cv.adaptiveThreshold(g_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 3)
    
    imagesToPhalanx = [image,bw_image,at_img]
    cv.imshow("Result",ImagePhalanx(RescaleAllImagesToHeight(imagesToPhalanx, 500), 2))
    cv.waitKey(0)
    cv.destroyAllWindows()
    
