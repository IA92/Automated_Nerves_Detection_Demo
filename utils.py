#Common Utility Functions
import os, cv2, csv, random, time
from pathlib import Path
from os import listdir
from os.path import isfile, join
import openslide
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing, binary_opening, binary_dilation, binary_erosion
from skimage.color import rgb2hsv
from skimage.transform import resize
from PIL import Image, ImageDraw
import scipy.misc    
from itertools import combinations, permutations

#Reading functions from file
def getRowsFromCSV(filePath, reType='Dict'):
    """Extract row data from CSV file and store it in either dictionary or list    
    filePath : path to the file [path in string].
    reType   : return type ['Dict' or 'List']. 
    Returns: [fileID: x1, y1, w, h] or [x1, y1, w, h].
    """
    try:
        if reType=='Dict':
            fReturn={}
            f=open(filePath, "r")
            line=f.readline()
            while line != "":
                line = line.strip().split(",")
                fReturn[line[0]]=[int(float(i)) for i in line[1:]]
                line=f.readline()
            f.close()
        elif reType=='List':
            fReturn=[]
            f=open(filePath, "r")
            line=f.readline()
            while line != "":
                line = line.strip().split(",")
                fReturn.append([int(float(i)) for i in line[1:]])
                line=f.readline()
            f.close()
    except IOError:
        if reType=='Dict':
            fReturn={}
        elif reType=='List':
            fReturn=[]
    return fReturn 


#Drawing functions
def drawBoxOnImage(imgArray, offsetX, offsetY, sizeX, sizeY, outline=(0, 255, 0)):
    """Draw a box on an image array    
    imgArray       : image array.
    offsetX        : Offset on X axis.
    offsetY        : Offset on Y axis.
    sizeX          : Width of the box.
    sizeY          : Height of the box. 
    Returns        : image with box [Image].
    """
    im=Image.fromarray(np.array(imgArray,'uint8'))
    draw = ImageDraw.Draw(im)
    draw.rectangle((offsetX, offsetY, offsetX+sizeX, offsetY+sizeY),  outline=outline)
    return im


def drawPredBoxOnImage(imgArray,imgArrayOffset, predsArrayInstance,  outline=(0, 255, 0)):
    """Draw a pred box on an image array.    
    imgArray              : image array.
    imgArrayOffset        : image X, Y coordinate on WSI [x,y].
    predsArrayInstance    : prediction box X,Y coordinate on WSI, width and height [x,y,w,h].
    Returns               : image with box [Image].
    """  
    offsetX=max([imgArrayOffset[0],predsArrayInstance[0]])-min([imgArrayOffset[0],predsArrayInstance[0]])
    offsetY=max([imgArrayOffset[1],predsArrayInstance[1]])-min([imgArrayOffset[1],predsArrayInstance[1]])
    sizeX=predsArrayInstance[2]
    sizeY=predsArrayInstance[3]    
    if (predsArrayInstance[0]-imgArrayOffset[0]) > 0 and (predsArrayInstance[1]-imgArrayOffset[1])>0:
        return drawBoxOnImage(imgArray,offsetX,offsetY,sizeX,sizeY,  outline=outline)
    else:
        return Image.fromarray(np.array(imgArray,'uint8'))

#Saving functions
def saveImageToJpg (fileDir, fileName, image, imageType='.jpg'):
    """Saving image to jpg file 
    fileDir     : path to the directory file [path in string].
    fileName    : file name in the directory file [string]. 
    image       : image [Image]
    """
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
    scipy.misc.imsave(os.path.join(fileDir,fileName+'.jpg'), image)
    return 1
    
def savingCSVFile (fileDir, fileName, inputList):
    """Saving list to CSV file fileName.csv located in fileDir folder 
    inputList  : input list to be saved to the CSV file [list of string].
    fileDir     : path to the directory file [path in string].
    fileName    : file name in the directory file [string]. 
    """
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
    fileDir= os.path.join(fileDir,fileName +'.csv')
    fPred=open(fileDir,'w', newline='');wr=csv.writer(fPred, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(inputList)):
        wr.writerow(inputList[i])
    print(str(len(inputList))+" rows stored successfully to "+fileDir)
    fPred.close() 
    return 1

def saveImageFromRoiPatchesWBoxToFolder (filePath, fileDir, fileName, patchesSet, roiList, randomOpt=True, patchesSize=(160,160), overlapIdx=True, pxlToUm=0.5036
 ):
    """saving image of ROI with box to folder  
    filePath    : path to the folder [path in string].
    fileDir     : folder name [string]. 
    fileName    : file name in the directory file [string].
    patchesSet  : set of image patches that belongs to N number of ROIs [N, [[imgArray],[lblArray],[xIdx,yIdx],[x,y]]].
    roiList     : list of roi boxes [N,[x,y,w,h]].
    randomOpt   : option to save images randomly or in sequence.
    patchesSize : size of the patches in image list[tupple of w,h].
    overlapIdx  : overlap option of how the image patches was extracted [True or False].
    pxlToUm     : conversion of pixel to Um (to be recorded in the output).
    Returns     : list of saved images information [x,y,w,h,x(Um),y(Um),fileDir,savingIdx] .
    """
    #Creating a list to store saved image information
    savedList=[]
    #Determining saving index, whether it is random or in sequence
    if len(patchesSet)>0:
        savingSeq=np.random.choice(len(patchesSet),len(patchesSet),replace=False) if randomOpt else range(len(patchesSet))
        #Determining sliding window size
        sWindowY=int(patchesSize[0]*0.5) if overlapIdx else int(patchesSize[0]);sWindowX=int(patchesSize[1]*0.5) if overlapIdx else int(patchesSize[1])
        #Setting saving index
        savingIdx=0
        for listIdx in savingSeq:
            imageOri=np.zeros((np.array(patchesSet[listIdx][2]).max(0)[1]*sWindowY+patchesSize[0],np.array(patchesSet[listIdx][2]).max(0)[0]*sWindowX+patchesSize[1],3))    
            for idx in range(len(patchesSet[listIdx][0])):
                imageOri[np.array(patchesSet[listIdx][2])[idx][1]*sWindowY:np.array(patchesSet[listIdx][2])[idx][1]*sWindowY+patchesSize[0],np.array(patchesSet[listIdx][2])[idx][0]*sWindowX:np.array(patchesSet[listIdx][2])[idx][0]*sWindowX+patchesSize[1],:]=np.array(patchesSet[listIdx][0][idx])
            savingIdx=savingIdx+1
            sampleFolderPath=os.path.join(filePath,fileDir,fileName)
            saveImageToJpg(sampleFolderPath,'Prediction_'+str(savingIdx),drawPredBoxOnImage(imageOri,patchesSet[listIdx][3][0],roiList[listIdx]))
            #storing info [x,y,w,h,x(um),y(um),folder,savingIndex]
            savedList.append([str(x) for x in [roiList[listIdx][0],roiList[listIdx][1],roiList[listIdx][2],roiList[listIdx][3],roiList[listIdx][0]*pxlToUm,roiList[listIdx][1]*pxlToUm,fileDir,savingIdx]])
    print(str(len(savedList))+" images saved successfully to "+os.path.join(fileDir,fileName))
    return savedList 
 



#Scoring functions
def boxOverlapCheck(boxA,boxListB,sizeTh=0):
    """Perform box overlapping check    
    boxA          : box [x,y,w,h].
    boxListB      : List of boxes. 
    sizeTh         : Overlapping size threshold [0 to 1].
    Returns        : The overlapping size index in boxListB.
    """
    boxAWMin=boxA[0];boxAWMax=boxA[0]+boxA[2]
    boxAHMin=boxA[1];boxAHMax=boxA[1]+boxA[3]
    boxListBidx=0
    for boxB in boxListB:
        boxBWMin=boxB[0];boxBWMax=boxB[0]+boxB[2]
        boxBHMin=boxB[1];boxBHMax=boxB[1]+boxB[3]
        dx = min(boxBWMax, boxAWMax) - max(boxBWMin, boxAWMin)
        dy = min(boxBHMax, boxAHMax) - max(boxBHMin, boxAHMin)
        if (dx>=0) and (dy>=0) and ((dx*dy>=sizeTh*boxA[2]*boxA[3]) or (dx*dy>=sizeTh*boxB[2]*boxB[3])):    
            return boxListBidx
        boxListBidx=boxListBidx+1
    else :
        return -1 

def scoringRoiWrtAnnot (rois, annotList, scoring=True):
    """Performing scoring of ROI w.r.t. annotations    
    rois         : list of roi bounding boxes [x,y,w,h].
    annotList    : list of annotation bounding boxes [x,y,w,h].
    Returns      : Number of annotations found, total number of annotations and intersected rois [int, int, [x,y,w,h]].
    """ 
    #defining intersection flag, intersection score, intersected ROI (True) [x,y,w,h] or unintersected ROI (False) [x,y,w,h]
    intFlag=False;intScore=[];trueROI=[];falseROI=[]
    #getting the number of annotations
    annotN=len(annotList)
    for roi in rois:
        roiWMin=roi[0];roiWMax=roi[0]+roi[2]
        roiHMin=roi[1];roiHMax=roi[1]+roi[3]
        #defining a dictionary to store the annotation bounding boxes with the amount of overlap as the keys
        annotScore={}
        for annot in annotList:
            annotWMin=annot[0];annotWMax=annot[0]+annot[2]
            annotHMin=annot[1];annotHMax=annot[1]+annot[3]
            dx = min(annotWMax, roiWMax) - max(annotWMin, roiWMin)
            dy = min(annotHMax, roiHMax) - max(annotHMin, roiHMin)
            if (dx>=0) and (dy>=0):
                intFlag=True
                if scoring:
                    #record the annot in a dictionary
                    annotScore[dx*dy+np.ceil((dx*dy)/(annot[2]*annot[3])*100)]=annot
        if intFlag:
            if scoring:
                #remove the annot with biggest intersection with the roi
                annotList.remove(annotScore[sorted(annotScore.keys(),reverse=True)[0]])
            #if there is intersection with any of the annotations, the roi is scored as 1 and the roi is recorded in trueROI list
            intScore.append(True)
            trueROI.append(roi)
            #reset flag
            intFlag=False
        else:
            #if no intersection with any of the annotations, the roi is scored as 0 and the roi is recorded in falseROI list
            intScore.append(False)
            falseROI.append(roi)
#    if scoring:
#        #print("Intersect "+str(np.array(intScore).sum())+" out of "+str(annotN))
#    else: 
#        #print("Finish categorising ROI to true or false ROI list")
    return (np.array(intScore).sum(),annotN,trueROI,falseROI)



#ROI Extraction functions
def filterColourRange(image, lLim, hLim, show=False):
    """Performing colour filter operation with a lower and upper limit.    
    image        : input immage array.
    lLim         : lower limit.
    hLim         : upper limit.
    show         : Display option [True or False].
    Returns      : Binary image array.
    """ 
    imCh = image.shape[-1]
    lLim = np.array(lLim).reshape((1,1,imCh))
    hLim = np.array(hLim).reshape((1,1,imCh))
    lImage = np.logical_and.reduce(image>lLim,-1)
    hImage = np.logical_and.reduce(image<hLim,-1)
    hlImage =  np.array(np.logical_and(lImage,hImage),'uint8')
    if show:
        plt.figure();plt.imshow(image)            
        plt.figure();plt.imshow(hlImage,'gray')
    return hlImage

def extractBBWithSizeTh(inputIm,areaTH=0):
    """Extract boxes with size threshold and convert it to bounding box   
    inputArray   : binary image.
    areaTH       : area threshold. 
    Returns  : list of contours that exceed the area threshold [x,y,w,h].
    """
    outputArr=[]
    _,ctrs, hier = cv2.findContours(np.array(inputIm,'uint8').copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(ctrs)):
        if ctrs[i].shape[0]>0:
            x,y,w,h=cv2.boundingRect(ctrs[i])
            if w*h>areaTH:
                outputArr.append([x,y,w,h])  
    return outputArr

def extractCtrsWithSizeTh(ctrs,areaTH=0):
    """Extract contours with size threshold   
    ctrs     : contours from cv2.findContours.
    areaTH   : area threshold. 
    Returns  : list of contours that exceed the area threshold.
    """
    uCtrs=[]
    for i in range(len(ctrs)):
        if ctrs[i].shape[0]>0:
            x,y,w,h=cv2.boundingRect(ctrs[i])
            if w*h>areaTH:
                uCtrs.append(ctrs[i])
    return uCtrs

def boundingBoxOnWSI (inputArray, offsetW=0,offsetH=0,offsetSW=0,offsetSH=0,dsF=1):
    """Getting the bounding boxes in WSI coordinate.   
    inputArray     : list of bounding boxes [x,y,w,h].
    offsetW/H      : offset of the box to be drawn on the image patch. 
    offsetSW/H     : offset of the box to be drawn on the WSI.
    dsF            : downsampling factor in WSI.
    Returns        : list of the bounding boxes coordinate in WSI original size.
    """
    outputArray=np.array(inputArray).copy()
    outputArray[:,0]=np.array(offsetSW+(offsetW+np.array(inputArray)[:,0])*dsF,int)
    outputArray[:,1]=np.array(offsetSH+(offsetH+np.array(inputArray)[:,1])*dsF,int)
    outputArray[:,2]=np.array(np.ceil(np.array(inputArray)[:,2]*dsF),int)
    outputArray[:,3]=np.array(np.ceil(np.array(inputArray)[:,3]*dsF),int)
    return outputArray






#Postprocessing functions
def paddedMorphologicalOperation (inputImg, opType='Close', filterSize=1):
    """Perform padded morphological closing or opening operation on the input image    
    inputImg     : input image [binary array].
    opType         : operation type ['Open' or 'Close']. 
    filterSize   : filter size for the morphological operation 
    Returns      : results from the operation [binary array with the same size of input image].
    """
    paddingSize=filterSize
    inputImgSizeW=np.array(inputImg).shape[1];inputImgSizeH=np.array(inputImg).shape[0]
    paddedImg=np.zeros((inputImgSizeH+paddingSize*2,inputImgSizeW+paddingSize*2))
    paddedImg[paddingSize:paddingSize+inputImgSizeH,paddingSize:paddingSize+inputImgSizeW]=inputImg
    if opType=='Close':
        paddedImg=binary_closing(np.array(paddedImg,'uint8'),np.ones((filterSize,filterSize)))
    elif opType=='Open':
        paddedImg=binary_opening(np.array(paddedImg,'uint8'),np.ones((filterSize,filterSize)))
    return paddedImg[paddingSize:paddingSize+inputImgSizeH,paddingSize:paddingSize+inputImgSizeW]

def processCC (inputArr, sizeTH=10,loc=False):
    """Getting either the binary image or location of the positive structure that is bigger than size threshold  
    inputArr : input image array.
    sizeTH   : size threshold. 
    loc      : option to output location (true) or binary image (false) of the positive structure.
    Returns  : list of location or binary image containing positive structure that exceed the threshold.
    """
    locList=[]
    #Getting the amount of connected components and image with label assigned to it (cCompN, cCompIm)
    (cCompN,cCompIm)=cv2.connectedComponents(np.array(inputArr,'uint8'))
    for i in range(cCompN-1):
        cCompTemp=np.array(cCompIm==i+1)
        #delete the locList is connected component is smaller than sizeTH 
        #else storing locList bounding boxes 
        if cCompTemp.sum()<sizeTH:
            cCompWhere=np.where(cCompTemp==True)
            for j in range(len(cCompWhere[0])):
                inputArr[cCompWhere[0][j],cCompWhere[1][j]]=False
        else:
            _,ctrs, hier = cv2.findContours(np.array(cCompTemp,'uint8').copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h=cv2.boundingRect(ctrs[0])
            locList.append([x,y,w,h])
    return locList if loc else inputArr 

def combineBoxes (inputArray, sizeTH=0):
    """Combining boxes in inputArray when it overlaps more than a size threshold    
    inputArray   : list of bounding boxes [x,y,w,h].
    sizeTh       : Overlapping size threshold.
    Returns      : list of bounding boxes resulted from combination operation [x,y,w,h].
    """ 
    outputArray=[]
    while len(inputArray)>0:   
        currentBox=inputArray[0]
        inputArray.pop(0)
        if len(inputArray)>0:
            boxMatchedIdx=boxOverlapCheck(currentBox,inputArray,sizeTH)
            while set(currentBox)==set(inputArray[boxMatchedIdx]):
                currentBox=inputArray[boxMatchedIdx]
                inputArray.pop(boxMatchedIdx)
                boxMatchedIdx=boxOverlapCheck(currentBox,inputArray,sizeTH)  
                if boxMatchedIdx==-1:
                    break
            if boxMatchedIdx>-1:
                boxMatched=inputArray[boxMatchedIdx]
                boxCombXMin=np.min([currentBox[0],boxMatched[0]])
                boxCombYMin=np.min([currentBox[1],boxMatched[1]])
                boxCombXMax=np.max([currentBox[0]+currentBox[2],boxMatched[0]+boxMatched[2]])
                boxCombYMax=np.max([currentBox[1]+currentBox[3],boxMatched[1]+boxMatched[3]])
                inputArray.pop(boxMatchedIdx)
                inputArray.append(np.array([boxCombXMin,boxCombYMin,boxCombXMax-boxCombXMin,boxCombYMax-boxCombYMin]))
            else:
                boxMatchedIdx=boxOverlapCheck(currentBox,outputArray,sizeTH)
                while boxMatchedIdx>-1:
                    outputArray.pop(boxMatchedIdx)
                    boxMatchedIdx=boxOverlapCheck(currentBox,outputArray,sizeTH)
                outputArray.append(currentBox)

        else:
                outputArray.append(currentBox) 
    return outputArray    

