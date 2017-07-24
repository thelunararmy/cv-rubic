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


# DEFINES
MaxDist2BlackToWhite = 765 # (255-0)^2 + (255-0)^2 + (255-0)^2


class ManualCropper():
    def __init__(self, prevImg, sampleNames, minBoxArea=100):
        # Set the starting params
        self.returnPoints = []
        self.cropping = False
        self.currentStartPoint = None
        self.currentMousePoint = None
        self.originalPreview = prevImg.copy()
        self.desiredSnippets = len(sampleNames)
        self.minBoxArea = minBoxArea
        self.sampleNames = sampleNames

    def GetPoints(self):
        # Create a named window 
        self.wname = "Create snippets"
        cv.namedWindow(self.wname)
        # Subscribe mouse clicks to that windows
        cv.setMouseCallback(self.wname, self.ClickCrop)
        
        # Setup resulting img
        resultingImg = None
        readyToContinue = False
        while True:
            # Clean the show img
            resultingImg = self.originalPreview.copy()
            
            # Draw boxes of each stored boxes
            for idx,(tl,br) in enumerate(self.returnPoints):
                cv.rectangle(resultingImg,tl,br,(255,255,255),3)
                cv.rectangle(resultingImg,tl,br,(0,80,0),2)
                cv.putText(resultingImg,"%s"%(self.sampleNames[idx]),(tl[0]+10,br[1]-10),cv.FONT_ITALIC,0.3,(255,255,255)) 
                
            # Draw box from start point to mouse pointer
            if self.currentStartPoint and self.currentMousePoint and self.cropping:
                # Get proper shape
                tl,br = self.ConvertToSquare(self.currentStartPoint,self.currentMousePoint)           
                if self.ValidateBox(tl,br):
                    cv.rectangle(resultingImg,tl,br,(100,255,100),2)
                else:
                    cv.rectangle(resultingImg,tl,br,(100,100,255),2)
                    
            # If the number of points has been reached then stop
            if len(self.returnPoints) >= self.desiredSnippets:
                cv.putText(resultingImg,"Samples Ready, Press any key to continue..." ,(10,resultingImg.shape[0]-10),cv.FONT_ITALIC,0.3,(255,255,255)) 
                readyToContinue = True
                
                        
            # Display the image and wait for a keypress
            cv.imshow(self.wname,resultingImg)
            
            # If the key is Q then kill
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                print "# Closing Manual Snipper"
                cv.destroyAllWindows()
                quit()
            elif key == ord("r"):
                print "# Resetting all snippets"
                readyToContinue = False
                del self.returnPoints[:]
            elif key == ord("u"):
                if self.returnPoints:
                    print "# Undoing last snippet"
                    readyToContinue = False
                    del self.returnPoints[-1]   
            elif key != 255 and readyToContinue:
                break
            
        # Destroy window
        cv.destroyWindow(self.wname)
        
        # Return our vals
        return self.returnPoints[:self.desiredSnippets]

    # Get the top left and bottom right corner of any given coordinates
    def BoxTLBR (self, (x1,y1), (x2,y2)):
        return (min(x1,x2),min(y1,y2)),(max(x1,x2),max(y1,y2))
    
    # Convert points to a perfect square
    def ConvertToSquare(self, tl, br):
        tl,br = (x1,y1),(x2,y2) = self.BoxTLBR (tl,br)
        # Determine box shape
        w = x2-x1
        h = y2-y1
        # Find smaller
        smallest = min(w,h)
        # Determine new BR
        x2 = x1 + smallest
        y2 = y1 + smallest  
        br = (x2,y2)           
        return tl,br
    
    # Ensure the given box isn't too rectangular
    def ValidateBox (self, (x1,y1), (x2,y2), verbose=False):
        # Determine if the box is useful
        # Check to see its cross section isnt minute
        if ( (x2-x1)**2 + (y2-y1)**2 ) **0.5 < self.minBoxArea:
            if verbose: print "# Snippet too small!"
            return False
        scale = 0.6
        if ( abs(x2-x1) * scale > abs(y2-y1) ):
            if verbose: print "# Snippet's width is rectangular!"
            return False
        if ( abs(y2-y1) * scale > abs(x2-x1) ):
            if verbose: print "# Snippet's height is rectangular!"
            return False
        # Everything else is okay!
        return True
    
    # Mouse click handler
    def ClickCrop(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv.EVENT_LBUTTONDOWN:
            if not self.cropping :
                # Record the start point and enable cropping on
                self.currentStartPoint = (x,y)
                self.cropping = True
            
        elif event == cv.EVENT_LBUTTONUP:
            if self.cropping and (len(self.returnPoints) < self.desiredSnippets) :
                tl,br = self.ConvertToSquare(self.currentStartPoint,(x,y))  
                #Ensure it is at least a feasible size
                if ( self.ValidateBox (tl,br,True) ) :
                    self.returnPoints.append( (tl,br) )                    
                
            # Reset everything to off
            self.currentStartPoint = None
            self.currentMousePoint = None
            self.cropping = False
        
        if self.cropping and event == cv.EVENT_MOUSEMOVE:
            # Handle mouse movements while in cropping mode
            self.currentMousePoint = (x,y)
        

def FetchImages ():
    '''
    Fetches all the file names within the Images folder
    '''
    fns = glob.glob("Images/*.jpg")
    imgs = [cv.imread(fn) for fn in fns]
    return imgs


def StitchSample(originalImage,colours):
    h,w,_ = originalImage.shape
    results = []
    emptyImg = np.zeros((h,w,3),np.uint8)
    emptyImg[:] = np.uint8(colours)
    results.append(emptyImg)
    return ImagePhalanx([originalImage]+results, 1)

def StitchExtractedResult(sampleName, (resName,resVal), perc ):
    emptyBox = np.zeros((15,25,3),np.uint8)
    emptyBox[:] = resVal
    emptyRect = np.zeros((15,300,3),np.uint8)
    cv.putText(emptyRect,"= %s, Proximity: %.2f"%(resName,perc),(3,13),cv.FONT_HERSHEY_PLAIN,0.8,(255,255,255))
    return ImageNaming(ImagePhalanx([emptyBox,emptyRect],1), "Best Result for '%s'"%(sampleName), 0.9)
     


CombiInOrder = ["Blood Ery","Urobilinogen","Bilirubin","Protein","Nitrite","Ketones","Glucose","pH","Specific Gravity","Leucocytes"]
CombiResults = {
                "Blood Ery" :     [
                                   ('neg',    [105,237,250]),
                                   ('ca. 10', [ 86,150,132]),
                                   ('ca. 50', [ 39, 86, 41]),
                                   ('ca. 250',[ 32, 38,  6])
                                  ],
                "Urobilinogen" :  [
                                   ('norm',   [151,163,179]),
                                   ('2',      [138,146,175]),
                                   ('4',      [127,128,178]),
                                   ('8',      [122,121,179]),
                                   ('12',     [108,100,181])
                                  ],
                "Bilirubin" :    [
                                  ('neg',     [144,170,186]),
                                  ('+',       [140,160,188]),
                                  ('++',      [124,146,185]),
                                  ('+++',     [112,126,179])
                                  ],
                "Protein" :      [
                                  ('neg',     [130,177,188]),
                                  ('30',      [126,164,163]),
                                  ('100',     [112,147,141]),
                                  ('500',     [104,128,111])
                                  ],
                "Nitrite" :      [
                                  ('neg',     [220,220,220]),
                                  ('pos',     [174,158,197])
                                  ],
                "Ketones" :      [
                                  ('neg',     [200,200,200]),
                                  ('+',       [156,130,160]),
                                  ('++',      [126, 77,128]),
                                  ('+++',     [ 83, 25, 72])
                                  ],
                "Glucose" :      [
                                  ('neg',     [155,212,210]),
                                  ('norm',    [126,166,136]),
                                  ('50',      [ 83,115, 69]),
                                  ('150',     [ 75, 67, 17]),
                                  ('500',     [ 52, 44, 17]),
                                  ('>1000',   [ 25, 17,  5])
                                 ],
                "pH" :           [
                                  ('5',       [114,135,203]),
                                  ('6',       [108,157,171]),
                                  ('7',       [ 68,127, 82]),
                                  ('8',       [ 72, 95, 30]),
                                  ('9',       [ 66, 50,  7]),
                                  ],
                "Specific Gravity" : [
                                  ('1.000',   [ 95, 67, 17]),
                                  ('1.005',   [ 76, 96, 62]),
                                  ('1.010',   [ 58, 90, 78]),
                                  ('1.015',   [ 63,107,100]),
                                  ('1.020',   [ 85,135,127]),
                                  ('1.025',   [ 99,152,155]),
                                  ('1.030',   [127,174,179])
                                  ],
                "Leucocytes" :    [
                                   ('neg',    [219,227,205]),
                                   ('ca. 25', [203,207,196]),
                                   ('ca. 75', [182,166,169]),
                                   ('ca. 500',[162,133,140])
                                   ]
                }

def GenerateCombiCatalog():
    finalCatalog = []
    for key, value in zip(CombiInOrder,[CombiResults[x]for x in CombiInOrder]):
        keyResults = []
        for name,col in value:
            emptyRect = np.zeros((15,60,3),np.uint8)
            cv.putText(emptyRect,name,(3,13),cv.FONT_HERSHEY_PLAIN,0.8,(255,255,255))
            emptySquare = np.zeros((15,50,3),np.uint8)
            emptySquare[:] = col
            keyResults.append(ImagePhalanx([emptyRect,emptySquare], 1))
        finalCatalog.append( ImageNaming(ImageVertigo(keyResults, 1), key, 0.8) )
    return ImageCatalog(finalCatalog,0,5)   

def GetBestResultForSample(sampleName,sampleCol):
    if CombiResults.has_key(sampleName) == False:
        raise ValueError("Cannot find sample name: ",sampleName)
    
    values = CombiResults[sampleName]
    cols = [np.float32(x) for _,x in values]
    scol32fc2 = np.float32(sampleCol)

    closestMatch = (9999.,-1) # eucl_val, index
    for idx,c in enumerate(cols):
        eucl = sum(cv.absdiff(c,scol32fc2))
        if closestMatch[0] >= eucl:
            closestMatch = (eucl,idx)
#         print c, eucl, cv.absdiff(c,scol32fc2)
    cEucl,index = closestMatch
    return sampleName, values[index], (int(cEucl) * 1.0 / MaxDist2BlackToWhite)

    
if __name__ == '__main__':
    image = FetchImages()[1]
    image = RescaleImageToHeight(image, 600)

    s = ManualCropper(image,CombiInOrder,25)
    boxPositions = s.GetPoints()
    del s
    
    extractedSamples = [image[y1:y2,x1:x2] for (x1,y1),(x2,y2) in boxPositions]
    
    resultingKColour = []
    for eS in extractedSamples:
        data32fc2 = eS.reshape((-1,3))
        data32fc2 = np.float32(data32fc2)
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,_,center = cv.kmeans(data32fc2,1,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
                
        center = np.uint8(center) 
        resultingKColour.append(center.tolist()[0])
    
    extractedSamples = RescaleAllImagesToHeight(extractedSamples, 50)
    extractedResult = []
    
    for idx,col in enumerate(resultingKColour):
        extractedResult.append(GetBestResultForSample(CombiInOrder[idx],col))
        
    displayKmeans = ImageVertigo([ StitchSample(es,rkc) for (es,rkc) in zip(extractedSamples,resultingKColour)],3)
    displayResults = ImageVertigo([ StitchExtractedResult(x,y,z) for x,y,z in extractedResult],4)
    
    talliedDisplay = ImagePhalanx([image,ImageNaming(displayKmeans,"RAW & EXT"),ImageVertigo([GenerateCombiCatalog(),displayResults])],8)
    DebugPointer(talliedDisplay)
    
     