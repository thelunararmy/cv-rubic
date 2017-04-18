import cv2 as cv
import numpy as np

def ImagePhalanx (images,linewidthbetween=0,bgcol=(0,0,0)):
    if len (images) <= 0:
        print "ImageVertigo was given an empty list!"
        return np.zeros((1,1,3),np.uint8)
    else:
        maxHeight = max([i.shape[0] for i in images])
        totalWidth = sum([i.shape[1] for i in images]) + linewidthbetween * (len(images) - 1) 
        returnImage = np.zeros((maxHeight,totalWidth,3),np.uint8)
        returnImage[:] = bgcol
        runningWidth = 0
        for i in range(len(images)):
            img = images[i] if images[i].ndim == 3 else cv.cvtColor(images[i],cv.COLOR_GRAY2BGR)
            currH, currW = img.shape[:2]
            returnImage[0:currH,runningWidth:runningWidth+currW] = img
            runningWidth += currW + linewidthbetween
        return returnImage

def ImageVertigo (images,linewidthbetween=0,bgcol=(0,0,0)):
    if len (images) <= 0:
        print "ImageVertigo was given an empty list!"
        return np.zeros((1,1,3),np.uint8)
    else:
        maxWidth = max([i.shape[1] for i in images])
        totalHeight = sum([i.shape[0] for i in images]) + linewidthbetween * (len(images) - 1) 
        returnImage = np.zeros((totalHeight,maxWidth,3),np.uint8)
        returnImage[:] = bgcol
        runningHeight = 0
        for i in range(len(images)):
            img = images[i] if images[i].ndim == 3 else cv.cvtColor(images[i],cv.COLOR_GRAY2BGR)
            currH, currW = img.shape[:2]
            returnImage[runningHeight:runningHeight+currH,0:currW] = img
            runningHeight += currH + linewidthbetween
        return returnImage
    
def ImageCatalog (images,imagesPerRow= -1,linewidthbetween=0,bgcol=(0,0,0)):
    if len (images) <= 0:
        print "ImageCatalog was given an empty list!"
        return np.zeros((1,1,3),np.uint8)
    else:
        imagesPerRow = int( len (images) / 2 ) if (imagesPerRow < 0) else imagesPerRow
        finalToBeVertigo = []
        runningPool = []
        for idx,img in enumerate(images):
            if ( idx == (len(images)-1) ) or ( (idx+1) % imagesPerRow == 0 ):
                runningPool.append(img)
                finalToBeVertigo.append(ImagePhalanx(runningPool, linewidthbetween, bgcol))
                del runningPool [:]
            else:
                runningPool.append(img)
        return ImageVertigo(finalToBeVertigo, linewidthbetween, bgcol)  

def ImageNaming (image, text, text_scale= .5,color = (255,255,255),bgcol =(0,0,0)):
    boxpadding = (20.0) * (text_scale / .5)
    h, w = image.shape[:2]
    returnImage = np.zeros((h+boxpadding,w,3),np.uint8)
    returnImage[:] = bgcol
    img = image if image.ndim == 3 else cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    returnImage[boxpadding:h+boxpadding,0:w] = img
    (tw,_),_ = cv.getTextSize(text, cv.FONT_ITALIC, text_scale, 1)
    cv.putText(returnImage,text,(int(returnImage.shape[1]/2.0 - tw/2.0),int(boxpadding*0.75)),cv.FONT_ITALIC,text_scale,color) 
    return returnImage

def NamedImageCatalog (imagesWithNames,imagesPerRow= -1,linewidthbetween=0,bgcol=(0,0,0),text_scale=1.0,textcol = (255,255,255) ):
    if len(imagesWithNames) == 0 or len(imagesWithNames[0]) != 2:
        print "NamedImageCatalog was either an empty imagesWithNames list or invalid pairings!"
        return np.zeros((1,1,3),np.uint8)
    else:
        toBeCataloged = []
        for image,name in imagesWithNames:
            toBeCataloged.append(ImageNaming(image,name,text_scale,textcol))
        return ImageCatalog(toBeCataloged, imagesPerRow, linewidthbetween, bgcol)
      
def RescaleImage (image,factor,method = cv.INTER_CUBIC ):
    return cv.resize(image,(int(image.shape[1]*factor*1.0),int(image.shape[0]*factor*1.0) ),0,0,interpolation=method)

def RescaleImageToHeight (image,height, method = cv.INTER_CUBIC):
    heightRatio = 1.0 * height / image.shape[0] 
    return cv.resize(image,(int(image.shape[1]*heightRatio*1.0),int(image.shape[0]*heightRatio*1.0) ),0,0,interpolation=method)

def RescaleImageToHeightWidth (image,height,width,method = cv.INTER_CUBIC):
    return cv.resize(image,(width,height),0,0,interpolation=method)

def RescaleImageToWidth (image,width, method = cv.INTER_CUBIC):
    widthRatio = 1.0 * width / image.shape[1] 
    return cv.resize(image,(int(image.shape[1]*widthRatio*1.0),int(image.shape[0]*widthRatio*1.0) ),0,0,interpolation=method)

def RescaleAllImagesToHeight(images,height,method = cv.INTER_CUBIC):
    return [RescaleImageToHeight(img, height,method) for img in images]