#Model predictions related functions
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

from utils import getRowsFromCSV,boundingBoxOnWSI,scoringRoiWrtAnnot,savingCSVFile
from utils import filterColourRange,paddedMorphologicalOperation,processCC,combineBoxes,saveImageFromRoiPatchesWBoxToFolder

#Model input preparation Functions
def roiToPatchesWithLabel(sliceDir, fileName, roi, patchesSize=(160,160), chN=3, overlapIdx=True):
    """Performing model input preparation from roi to patches with label.    
    sliceDir     : WSI folder path.
    fileName     : file name of the slice.
    roi          : roi bounding box [x,y,w,h].
    patchesSize   : size of image patches.
    chN          : Number of channels used in the image [int].
    overlapIdx   : 50% overlapping option for the image patches [True or False].
    Returns      : Array of image patches, label patches and patches X, Y index on full ROI image & coordinate on WSI [[imgArray],[lblArray],[xIdx,yIdx],[x,y]].
    """ 
    sliceFile=os.path.join(sliceDir,fileName+'.svs') 
    slide=openslide.open_slide(sliceFile)
    outputArray=[];outputLabel=[];outputIdx=[]    
    #Calculating how many image patches are required to cover the width and height of the ROI
    winWmultiplyer=int(np.ceil(roi[2]/patchesSize[0]));winHmultiplyer=int(np.ceil(roi[3]/patchesSize[1]))
    #Calculating the starting x, y coordinate where the image patches should be extracted from
    xBR=int(roi[0]-(patchesSize[0]*winWmultiplyer-roi[2])/2);yBR=int(roi[1]-(patchesSize[1]*winHmultiplyer-roi[3])/2)
    #Increasing the amount of image patches to extract from and reducing the sliding size by half, if the overlapping option == True
    if overlapIdx: winWmultiplyer=winWmultiplyer*2-1; winHmultiplyer=winHmultiplyer*2-1; slideSize=int(0.5*patchesSize[0])
    #Extracting image patches
    for widthIdx in range(winWmultiplyer):    
        for heightIdx in range(winHmultiplyer):
            #Reading the slide and getting the data
            temp=slide.read_region((xBR+widthIdx*slideSize,yBR+heightIdx*slideSize),0,patchesSize)   
            temp=np.array(temp.getdata(),'uint8').reshape(patchesSize[1],patchesSize[0], np.array(temp.getdata()).shape[-1])  
            #Converting the RGB image to HSV image and applying colour filter
            tempR=temp[:,:,:chN]
            tempL=rgb2hsv(np.array(tempR,'uint8'))  
            tempL=filterColourRange(tempL,[0.04,0.2,0.4],[0.2,1.,1.]) #Colour filter in HSV space
            #Appending to output list
            outputArray.append(np.array(tempR,'uint8'))  
            outputLabel.append(np.array(tempL,'uint8'))
            outputIdx.append([[widthIdx,heightIdx],[xBR+widthIdx*slideSize,yBR+heightIdx*slideSize]])
    return (outputArray,outputLabel,outputIdx)

def roiListToSetPatchesWithLabel (sliceDir, fileName, roiList, patchesSize=(160,160), chN=3, overlapIdx=True):
    """Performing model input preparation from list of roi to set of patches with label.    
    sliceDir     : WSI folder path.
    fileName     : file name of the slice.
    roiList      : list of roi bounding boxes [x,y,w,h].
    patchesSize   : size of image patches.
    chN          : Number of channels used in the image [int].
    overlapIdx   : 50% overlapping option for the image patches [True or False].
    Returns      : List of image patches, label patches and X, Y index on full ROI image & coordinate on WSI coordinate on WSI [[N,imgArray],[N,lblArray],[N,[xIdx,yIdx]],[N,[x,y]].
    """ 
    outputList=[]  
    for idx in range(len(roiList)):
        temp=roiToPatchesWithLabel(sliceDir, fileName, roiList[idx], patchesSize=patchesSize, chN=chN, overlapIdx=overlapIdx)
        outputList.append([temp[0],temp[1],np.array(temp[2])[:,0,:],np.array(temp[2])[:,1,:]])
    return outputList


#Predicting functions
def buildingImageFromRoiPatchesSet (patchesSet, pred=False, overlapIdx=True ):
    """building images form set of ROI patches  
    patchesSet   : list of set of image patches that belongs to N number of ROIs [N, [[imgArray],[lblArray],[xIdx,yIdx],[x,y]]].
    pred         : prediction option of whether the patches is RGB image or not [True or False].
    overlapIdx   : overlap option of how the image patches was extracted [True or False].
    Returns      : list of ROI images.
    """
    imagesList=[];patchesSize=np.array(patchesSet[0][0]).shape[1:3]
    patchesChn=0 if pred else np.array(patchesSet[0][0]).shape[3]
    #Determining sliding window size
    sWindowY=int(patchesSize[0]*0.5) if overlapIdx else int(patchesSize[0]);sWindowX=int(patchesSize[1]*0.5) if overlapIdx else int(patchesSize[1])
    for setIdx in range(len(patchesSet)):
        if pred:
            imageOri=np.zeros((np.array(patchesSet[setIdx][2]).max(0)[1]*sWindowY+patchesSize[0],np.array(patchesSet[setIdx][2]).max(0)[0]*sWindowX+patchesSize[1]))
            for idx in range(len(patchesSet[setIdx][0])):
                xIdx=patchesSet[setIdx][2][idx][0];yIdx=patchesSet[setIdx][2][idx][1]
                patch=np.logical_or(imageOri[yIdx*sWindowY:yIdx*sWindowY+patchesSize[0],xIdx*sWindowX:xIdx*sWindowX+patchesSize[1]],patchesSet[setIdx][0][idx])
                imageOri[yIdx*sWindowY:yIdx*sWindowY+patchesSize[0],xIdx*sWindowX:xIdx*sWindowX+patchesSize[1]]=patch
        else :
            imageOri=np.zeros((np.array(patchesSet[setIdx][2]).max(0)[1]*sWindowY+patchesSize[0],np.array(patchesSet[setIdx][2]).max(0)[0]*sWindowX+patchesSize[1],patchesChn))    
            for idx in range(len(patchesSet[setIdx][0])):
                xIdx=patchesSet[setIdx][2][idx][0];yIdx=patchesSet[setIdx][2][idx][1]
                patch=np.array(patchesSet[setIdx][0][idx])
                imageOri[yIdx*sWindowY:yIdx*sWindowY+patchesSize[0],xIdx*sWindowX:xIdx*sWindowX+patchesSize[1],:]=patch
        imagesList.append(np.array(imageOri,'uint8'))
    return imagesList 

def predPatchesSet (slicePath, fileName, dlB, patchesSet, xTrainMean, xTrainStd):
    """predicting image patches with model DL-B
    dlB          : model with dual output (segmentation and classification output)
    patchesSet   : list of set of image patches that belongs to N number of ROIs [N, [[imgArray],[lblArray],[xIdx,yIdx],[x,y]]].
    xTrainMean   : training set mean for normalisation.
    xTrainStd    : training set standard deviation for normalisation.
    Returns      : list of positive instances set of patches, instances location, number of instances, number of positive patches set.
    """
    predSet=[];predLoc=[];lenRoi=0
    for setIdx in range(len(patchesSet)):
        predIm=patchesSet.copy()
        for idx in range(len(patchesSet[setIdx][0])):
            predInit=dlB.predict(((np.array(patchesSet[setIdx][0][idx])-xTrainMean)/xTrainStd).reshape((-1,160,160,3)))
            segOut=processCC((predInit[0][0,:,:,0]>0.5),0)
            classOut=predInit[1][0]
            if classOut>0.5:  
                predIm[setIdx][0][idx]=segOut
            else :
                predIm[setIdx][0][idx]=np.zeros(segOut.shape)
        temp=buildingImageFromRoiPatchesSet ([predIm[setIdx]], pred=True, overlapIdx=True )[0]
        temp=paddedMorphologicalOperation(np.array(temp,'uint8'),'Close',20) 
        extractedBB=processCC(np.array(temp,'uint8'),100,loc=True)
        if len(extractedBB)>0:
            lenRoi=lenRoi+1
            predLoc.extend(boundingBoxOnWSI (extractedBB, offsetW=0,offsetH=0,offsetSW=patchesSet[setIdx][3][0][0],offsetSH=patchesSet[setIdx][3][0][1],dsF=1))
    predLoc=combineBoxes(predLoc, sizeTH=0.5);lenPreds=len(predLoc)
    predSet=roiListToSetPatchesWithLabel(slicePath, fileName, predLoc, patchesSize=(160,160), chN=3, overlapIdx=True)
    return (predSet,predLoc,lenPreds,lenRoi) 

def predictListWSI (slicePath, fileList, fileROIs, model, xTrainMean, xTrainStd, annotPath="", fileDir=""):
    """predict list of WSIs    
    slicePath   : WSI directory path [string].
    fileList    : list of file names to be processed [list of string].
    fileROIs    : dictionary of ROIs for each of the WSI in the list.
    model       : model with dual output (segmentation and classification output)
    xTrainMean   : training set mean for normalisation.
    xTrainStd    : training set standard deviation for normalisation.
    annotPath   : annotation csv directory to "AnnotFile xxxxx.csv" file. 
                  If none, will not process scoring.
                  If wrong path, will output 0 out of 0. 
    fileDir     : directory where "pred csv" folder will be created.
    Returns     : dictionary of testing scores for each of the WSI in the list.
    """ 
    testingScores={}
    for fileIdx in range(0,len(fileList)):
        fileName=fileList[fileIdx];toSaveList=[]
        if annotPath!="":
            #make sure to put "AnnotFile xxxxx.csv" in annotPath
            annotList=getRowsFromCSV(os.path.join(annotPath,'AnnotFile '+str(fileName)+'.csv'),'List')  
        xTrue,xFalse=scoringRoiWrtAnnot (fileROIs[fileName], annotList, scoring=False)[2:]
        
        xTrueSet=roiListToSetPatchesWithLabel(slicePath, fileName, xTrue, patchesSize=(160,160), chN=3, overlapIdx=True)
        predSet,predLoc,lenPredsAnnot,lenRoiAnnot=predPatchesSet(slicePath, fileName,model,xTrueSet,xTrainMean,xTrainStd)
        if fileDir!="": toSaveList.extend(saveImageFromRoiPatchesWBoxToFolder (os.path.join(fileDir,'pred csv','Model'), 'AnnotDetected', str(fileName), predSet, predLoc,randomOpt=True ))
        intScore,annotN=list(scoringRoiWrtAnnot (predLoc, annotList.copy())[:2])
            
        xFalseSet=roiListToSetPatchesWithLabel(slicePath, fileName, xFalse, patchesSize=(160,160), chN=3, overlapIdx=True)
        predSet,predLoc,lenPredsNonAnnot,lenRoiNonAnnot=predPatchesSet(slicePath, fileName,model,xFalseSet,xTrainMean,xTrainStd)
        if fileDir!="": toSaveList.extend(saveImageFromRoiPatchesWBoxToFolder (os.path.join(fileDir,'pred csv','Model'), 'NonAnnotDetected', str(fileName), predSet, predLoc,randomOpt=True ))
        
        testingScores[fileName]=[[intScore,annotN],[lenRoiNonAnnot,len(xFalse)],[lenPredsAnnot,lenPredsNonAnnot]] 
        if fileDir!="": savingCSVFile(os.path.join(fileDir,'pred csv','Model'),'PredFile '+str(fileName),toSaveList)
    return testingScores

def filterListWSI (slicePath, fileList, fileROIs, fileScores, annotPath="", fileDir=""):
    """filter list of WSIs    
    slicePath   : WSI directory path [string].
    fileList    : list of file names to be processed [list of string].
    fileROIs    : dictionary of ROIs for each of the WSI in the list.
    fileScores  : dictionary of scores for each of the WSI in the list.
    annotPath   : annotation csv directory to "AnnotFile xxxxx.csv" file. 
                  If none, will not process scoring.
                  If wrong path, will output 0 out of 0. 
    fileDir     : directory where "pred csv" folder will be created.
    Returns     : dictionary of testing scores for each of the WSI in the list.
    """ 
    testingScores={}
    for fileIdx in range(0,len(fileList)):
        fileName=fileList[fileIdx];toSaveList=[]
        if annotPath!="":
            #make sure to put "AnnotFile xxxxx.csv" in annotPath
            annotList=getRowsFromCSV(os.path.join(annotPath,'AnnotFile '+str(fileName)+'.csv'),'List')  
        xTrue,xFalse=scoringRoiWrtAnnot (fileROIs[fileName], annotList, scoring=False)[2:]
        
        xTrueSet=roiListToSetPatchesWithLabel(slicePath, fileName, xTrue, patchesSize=(160,160), chN=3, overlapIdx=True)
        if fileDir!="": toSaveList.extend(saveImageFromRoiPatchesWBoxToFolder (os.path.join(fileDir,'pred csv','CF'), 'AnnotDetected', str(fileName), xTrueSet, xTrue,randomOpt=True ))
            
        xFalseSet=roiListToSetPatchesWithLabel(slicePath, fileName, xFalse, patchesSize=(160,160), chN=3, overlapIdx=True)
        if fileDir!="": toSaveList.extend(saveImageFromRoiPatchesWBoxToFolder (os.path.join(fileDir,'pred csv','CF'), 'NonAnnotDetected', str(fileName), xFalseSet, xFalse,randomOpt=True ))
        
        testingScores[fileName]=[fileScores[fileName],[len(xTrue),len(xFalse)],[len(xTrueSet),len(xFalseSet)]] 
        if fileDir!="": savingCSVFile(os.path.join(fileDir,'pred csv','CF'),'RoiFile '+str(fileName),toSaveList)
    return testingScores

def printscoresDict (scoresDict):
    """print the testing scores of wsi
    scoresDict     : dictionary of scores results from predictListWSI function
    """
    print("Testing results :")
    for wsi in list(scoresDict.keys()):
        print(str(wsi)+" : Identified annotations "+str(scoresDict[wsi][0])+" and Additional "+str(scoresDict[wsi][2][1]))
