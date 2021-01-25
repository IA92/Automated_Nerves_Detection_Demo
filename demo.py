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

from utils import drawPredBoxOnImage,getRowsFromCSV,filterColourRange
from IPython.display import display

def demoFile (fileName, dArea, fileDir, slicePath, annotPath):
    """Visualise the original image, raw colour filter output, original image with experts annotations (boxes in red) and original image with predictions (boxes in green).    
    fileName    : name of the file to be processed [string].
    dArea       : slide area coordinate in the WSI to be processed for demo[x,y,w,h].
    fileDir     : directory where "pred csv" folder exists.
    slicePath   : WSI directory path [string].
    annotPath   : annotation csv directory to "AnnotFile xxxxx.csv" file.
    """
    predFilePath=os.path.join(fileDir,'pred csv','Model','PredFile '+str(fileName)+'.csv')
    predLoc=getRowsFromCSV_Demo(predFilePath)
    
    annotList=getRowsFromCSV(os.path.join(annotPath,'AnnotFile '+fileName+'.csv'),'List') 
    
    imDemo=returnSlideArea_Demo(slicePath, fileName, dArea[fileName])
    print("Drawing original image")
    display(Image.fromarray(np.array(imDemo,'uint8')))

    print("Colour filter output")
    cfOut=filterColourRange(rgb2hsv(np.array(imDemo[:,:,:3],'uint8')),[0.04,0.2,0.4],[0.2,1.,1.])
    display(Image.fromarray(cfOut*255))

    print("Drawing manual annotation by experts")
    display(drawListOfPredBoxes_Demo(imDemo, dArea[fileName], annotList,(255,0,0)))

    print("Drawing predictions by model")
    display(drawListOfPredBoxes_Demo(imDemo, dArea[fileName], predLoc,(0,255,0)))

def getRowsFromCSV_Demo(filePath):
    """Extract row data from column 0 to 4 from CSV file and store it in list    
    filePath : path to the file [path in string].
    Returns: [x, y, w, h].
    """
    try:
        fReturn=[]
        f=open(filePath, "r")
        line=f.readline()
        while line != "":
            line = line.strip().split(",")
            fReturn.append([int(float(i)) for i in line[0:4]])
            line=f.readline()
        f.close()
    except IOError:
        fReturn=[]
    return fReturn 

def returnSlideArea_Demo (slicePath, fileName, slideArea, sLevel=0, sDivider=1):
    """Return an image array as listed in the slide area in the original resolution for demo    
    slicePath   : WSI directory path [string].
    fileName    : name of the file to be processed [string].
    slideArea   : slide area coordinate in the WSI to be processed [x,y,w,h].
    sLevel      : Slide resolution level [0 - 4], for demo sLevel should equal to 0.
    sDivider    : The number of image blocks that the WSI is divided to [int], for demo sDivider should equal to 0.
    Returns     : image array.
    """
    #Opening WSI at full resolution, where dsF is the downsampling factor
    sliceFile=os.path.join(slicePath,fileName+'.svs')   
    slide=openslide.open_slide(sliceFile);dsF=slide.level_downsamples[0]
    #Getting the downsampled area of the slide to be processed
    sLevelW=int(np.ceil(slideArea[2]/dsF));sLevelH=int(np.ceil(slideArea[3]/dsF))
    #Getting the initial x, y coordinate of the area to be processed
    offsetSW=slideArea[0];offsetSH=slideArea[1]
    #Getting the area size of the image block/patch to be processed, which is w, h of the area divided by sDivider
    sPatchW=int(np.ceil(sLevelW/sDivider));sPatchH=int(np.ceil(sLevelH/sDivider))
    patchIdxH=0;offsetH=patchIdxH*sPatchH;patchIdxW=0;offsetW=patchIdxW*sPatchW
    temp=slide.read_region((int(offsetSW+offsetW*dsF),int(offsetSH+offsetH*dsF)),sLevel,(int(sPatchW),int(sPatchH)))
    temp=np.array(temp.getdata(),'uint8').reshape(sPatchH, sPatchW, np.array(temp.getdata()).shape[-1])
    return temp

def drawListOfPredBoxes_Demo (image, slideArea, predList,outline=(0, 255, 0)):
    """Draw a list of prediction boxes on an image array.    
    image                 : image or image array.
    slideArea             : slide area coordinate in the WSI to be processed [x,y,w,h].
    predList              : list of prediction box X,Y coordinate on WSI, width and height [x,y,w,h].
    outline               : colour value (R,G,B).
    Returns               : image with boxes [Image].
    """      
    for idx in range(len(predList)):
        image=drawPredBoxOnImage(image, slideArea, predList[idx],outline)
    return Image.fromarray(np.array(image,'uint8'))
