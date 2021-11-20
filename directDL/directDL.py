#Direct deep learning (DirectDL) functions
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

from utils import getRowsFromCSV,boundingBoxOnWSI
from utils import savingCSVFile, saveImageFromRoiPatchesWBoxToFolder

from utils import getRowsFromCSV,extractBBWithSizeTh,boundingBoxOnWSI,paddedMorphologicalOperation,boxOverlapCheck
from utils import filterColourRange,drawPredBoxOnImage
from utils import scoringRoiWrtAnnot,combineBoxes
from utils import processCC
from predict import roiListToSetPatchesWithLabel


def predictDirectDL(model, xTrainMean, xTrainStd, slicePath, fileName, sArea, sLevel=0, patchSize=160, overlap=0.5):
    """Extract roi directly from WSI with specified pretrained DL model, xTrainMean, xTrainStd, slide area, slide level, input patch size, overlap flag 
    model      : a deep learning model B.
    xTrainMean : training set mean for normalisation.
    xTrainStd  : training set standard deviation for normalisation.
    slicePath  : Path to the WSI directory.
    fileName   : Name of WSI. 
    sArea      : Dictionary of slide area.
    sLevel     : Slide resolution level [0 - 4].
    patchSize  : The size of width and height of the input image patches (in px) for the DL model [int].
    overlap    : Patch overlap size (0 or 0.5).
    Returns    : List of the bounding box rois [x, y, w, h]. 
    """
    rois=[]
    #Opening WSI at 4x downsampled resolution, where dsF is the downsampling factor
    sliceFile=os.path.join(slicePath,fileName+'.svs')   
    slide=openslide.open_slide(sliceFile);dsF=slide.level_downsamples[sLevel]
    #Getting slideArea bounding box coordinate [x, y, w, h]
    try:
        slideArea=sArea[fileName]
    except KeyError:
        slideArea=[0,0,slide.dimensions[0],slide.dimensions[1]]
    #Getting the number of slide divider by dividing the sArea by the patch size
    sDividerW=np.ceil(slideArea[2]/dsF/patchSize);sDividerH=np.ceil(slideArea[3]/dsF/patchSize) 
    #Getting the downsampled area of the slide to be processed
    sLevelW=int(np.ceil(slideArea[2]/dsF));sLevelH=int(np.ceil(slideArea[3]/dsF))
    #Getting the initial x, y coordinate of the area to be processed
    offsetSW=slideArea[0];offsetSH=slideArea[1]
    #Getting the area size of the image block/patch to be processed, which is w, h of the area divided by sDivider
    sPatchW=patchSize;sPatchH=patchSize
    #Processing each of the image block
    for patchIdxH in range(int(sDividerH/overlap)-1):
        offsetH=patchIdxH*sPatchH-patchIdxH*int(sPatchH*overlap)
        for patchIdxW in range(int(sDividerW/overlap)-1):
            offsetW=patchIdxW*sPatchW-patchIdxW*int(sPatchW*overlap) 
            #Obtaining image block as image array
            temp=slide.read_region((int(offsetSW+offsetW*dsF),int(offsetSH+offsetH*dsF)),sLevel,(int(sPatchW),int(sPatchH)))
            temp=np.array(temp.getdata(),'uint8').reshape(sPatchH, sPatchW, np.array(temp.getdata()).shape[-1])
            temp=temp[:,:,:3].copy()
            #Predict the patch with the DL model
            predInit=model.predict(((np.array(temp)-xTrainMean)/xTrainStd).reshape((-1,patchSize,patchSize,3)))
            segOut=processCC((predInit[0][0,:,:,0]>0.5),0)
            classOut=predInit[1][0]
            if classOut>0.5:  
                temp=segOut
                temp=paddedMorphologicalOperation(np.array(temp,'uint8'),'Close',20) 
                extractedBB=extractBBWithSizeTh(np.array(temp,'uint8'),areaTH=100)
                #extractedBB=processCC(np.array(temp,'uint8'),100,loc=True)           
                if len(extractedBB)>0:
                    rois.extend(boundingBoxOnWSI(extractedBB,offsetW,offsetH,offsetSW,offsetSH,dsF))
            else :
                temp=np.zeros(segOut.shape)
    return rois       
             
def directPredictListWSI (model, xTrainMean, xTrainStd, slicePath, fileList, annotPath="", sArea={}, sizeTH=0):
    """Direct predict list of WSIs   
    model      : a deep learning model B.
    xTrainMean : training set mean for normalisation.
    xTrainStd  : training set standard deviation for normalisation.
    slicePath   : WSI directory path [string].
    fileList    : list of file names to be processed [list of string].
    annotPath   : annotation csv directory to "AnnotFile xxxxx.csv" file. 
                  If none, will not process scoring.
                  If wrong path, will output 0 out of 0. 
    sArea       : Dictionary of area to be processed. If empty, will take the full dimension of the slide. 
    sizeTh      : Overlapping size threshold.
    Returns     : dictionary of ROIs and scores for each of the WSI in the list.
    """ 
    fileScores={}
    fileROIs={}
    for fileIdx in range(0,len(fileList)):
        rois=predictDirectDL(model, xTrainMean, xTrainStd, slicePath, fileList[fileIdx], sArea) 
        if annotPath!="":
            #make sure to put "AnnotFile xxxxx.csv" in annotPath
            annotList=getRowsFromCSV(os.path.join(annotPath,'AnnotFile '+fileList[fileIdx]+'.csv'),'List')  
            fileScores[fileList[fileIdx]]=list(scoringRoiWrtAnnot (rois, annotList.copy())[:2])
        fileROIs[fileList[fileIdx]]=combineBoxes(rois.copy(),sizeTH)
    return (fileROIs, fileScores)

def directDLResults (slicePath, fileList, fileROIs, fileScores, annotPath="", fileDir=""):
    """Scoring fileROIs that made of a list of WSIs    
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
        if fileDir!="": toSaveList.extend(saveImageFromRoiPatchesWBoxToFolder (os.path.join(fileDir,'pred csv','DirectDL'), 'AnnotDetected', str(fileName), xTrueSet, xTrue,randomOpt=True ))
            
        xFalseSet=roiListToSetPatchesWithLabel(slicePath, fileName, xFalse, patchesSize=(160,160), chN=3, overlapIdx=True)
        if fileDir!="": toSaveList.extend(saveImageFromRoiPatchesWBoxToFolder (os.path.join(fileDir,'pred csv','DirectDL'), 'NonAnnotDetected', str(fileName), xFalseSet, xFalse,randomOpt=True ))
        
        testingScores[fileName]=[fileScores[fileName],[len(xTrue),len(xFalse)],[len(xTrueSet),len(xFalseSet)]] 
        if fileDir!="": savingCSVFile(os.path.join(fileDir,'pred csv','DirectDL'),'DirectPredFile '+str(fileName),toSaveList)
    return testingScores