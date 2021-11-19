#Preprocessing functions
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

from utils import getRowsFromCSV,extractBBWithSizeTh,boundingBoxOnWSI,paddedMorphologicalOperation,boxOverlapCheck
from utils import filterColourRange,drawPredBoxOnImage
from utils import scoringRoiWrtAnnot,combineBoxes
from utils import processCC


def roiExtraction(slicePath, fileName, sArea, sLevel=1, sDivider=16, roiTH=100):
    """Extract roi from WSI with specified slide area, slide level, slide divider and size threshold of roi    
    slicePath  : Path to the WSI directory.
    fileName   : Name of WSI. 
    sArea      : Dictionary of slide area.
    sLevel     : Slide resolution level [0 - 4].
    sDivider   : The number of image blocks that the WSI is divided to [int].
    roiTH      : The size threshold (in px) of the ROI to be extracted [int].
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
    #Getting the downsampled area of the slide to be processed
    sLevelW=int(np.ceil(slideArea[2]/dsF));sLevelH=int(np.ceil(slideArea[3]/dsF))
    #Getting the initial x, y coordinate of the area to be processed
    offsetSW=slideArea[0];offsetSH=slideArea[1]
    #Getting the area size of the image block/patch to be processed, which is w, h of the area divided by sDivider
    sPatchW=int(np.ceil(sLevelW/sDivider));sPatchH=int(np.ceil(sLevelH/sDivider))
    #Processing each of the image block
    for patchIdxH in range(sDivider):
        offsetH=patchIdxH*sPatchH
        for patchIdxW in range(sDivider):
            offsetW=patchIdxW*sPatchW
            #Obtaining image block as image array
            temp=slide.read_region((int(offsetSW+offsetW*dsF),int(offsetSH+offsetH*dsF)),sLevel,(int(sPatchW),int(sPatchH)))
            temp=np.array(temp.getdata(),'uint8').reshape(sPatchH, sPatchW, np.array(temp.getdata()).shape[-1])
            #Apply colour filter in HSV space
            tempHSV=rgb2hsv(temp[:,:,:3])
            tempBinary=np.array(filterColourRange(tempHSV[:,:,:3],[0.04,0.2,0.4],[0.2,1.,1.]),'uint8')
            #Apply padded 20x20 pixel morphological closing, where the size of the filter is multiplied by dsF
            temp=paddedMorphologicalOperation(np.array(tempBinary,'uint8'),'Close',5)   
            temp=np.array(temp,'uint8')
            #Extract rois with minimum size of roi size threshold
            extractedBB=extractBBWithSizeTh(temp.copy(),areaTH=roiTH/(dsF*dsF))
            if len(extractedBB)>0:
                rois.extend(boundingBoxOnWSI(extractedBB,offsetW,offsetH,offsetSW,offsetSH,dsF))
    return rois       
             
def preprocessListWSI (slicePath, fileList, annotPath="", sArea={}, sizeTH=0):
    """Preprocess list of WSIs    
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
        rois=roiExtraction(slicePath, fileList[fileIdx], sArea) 
        if annotPath!="":
            #make sure to put "AnnotFile xxxxx.csv" in annotPath
            annotList=getRowsFromCSV(os.path.join(annotPath,'AnnotFile '+fileList[fileIdx]+'.csv'),'List')  
            fileScores[fileList[fileIdx]]=list(scoringRoiWrtAnnot (rois, annotList.copy())[:2])
        fileROIs[fileList[fileIdx]]=combineBoxes(rois.copy(),sizeTH)
    return (fileROIs, fileScores)
  
  
