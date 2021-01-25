import qupath.lib.objects.*
import qupath.lib.roi.*
import qupath.lib.scripting.QPEx
def imageData = QPEx.getCurrentImageData()

// Any vertices
def x = [0] as float[]
def y = [0] as float[]

def xa=0 as float
def xb=0 as float
def ya=0 as float
def yb=0 as float

// Create object
def roi = new PolygonROI(x, y, -1, 0, 0)
def pathObject = new PathAnnotationObject(roi)
def rgb = getColorRGB(0, 200, 0)
def pathClass = getPathClass('Prediction', rgb)

//Path to prediction file that contains rows of x,y,w,h 
def file = new File('D:/UniData/Chris Dataset/pred csv/PredFile'+imageData.toString().split(',')[-1]+'.csv')

// Index offset where the x,y,w,h columns start 
idxOffset=0
def lineNo = 1
      def line
      file.withReader { reader ->
         while ((line = reader.readLine())!=null) {
            println "${lineNo}. ${line.split(',')[1]}"
            xa=line.split(',')[0+idxOffset] as float
            xb=line.split(',')[2+idxOffset] as float
            x = [xa,xa,xa+xb,xa+xb] as float[]
            println "${lineNo}. ${x}"
            ya=line.split(',')[1+idxOffset] as float
            yb=line.split(',')[3+idxOffset] as float
            y = [ya,ya+yb,ya+yb,ya] as float[]
            roi = new PolygonROI(x, y, -1, 0, 0)
            pathObject = new PathAnnotationObject(roi, pathClass)
            addObject(pathObject)   
            lineNo++
         }
      }
           
