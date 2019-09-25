import sys
sys.path.append("..")
import numpy as np
import Helper
import math
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import minmax_scale
def createReferenceSample(sampleSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd , multipleBox=False):
    DataSet = np.zeros(( sampleSize[0],sampleSize[1]))

    if multipleBox:
        numOfBoxes=len(TargetYStart)
        for i in range(numOfBoxes):
            DataSet[TargetYStart[i]:TargetYEnd[i],TargetXStart[i]:TargetXEnd[i]]=1
    else:
            DataSet[TargetYStart:TargetYEnd,TargetXStart:TargetXEnd]=1
    return DataSet



def getNumberOfImportantFeatures( ImpTS , ImpFeatures , multipleBox=False):
    number=0
    if multipleBox:
        numOfBoxes=len(ImpTS)
        for i in range(numOfBoxes):
            number+=ImpTS[i]*ImpFeatures[i]      
    else:
            number+=ImpTS*ImpFeatures
    return number



def changeProbToClass(ImpFeaturesIndex,Sample):
    newSample=np.zeros((Sample.shape))
    newClass=np.zeros((Sample.shape))

    newSample[ImpFeaturesIndex]=Sample[ImpFeaturesIndex]
    newClass[ImpFeaturesIndex]=1
    return newClass,  newSample

def getBoxInfo(type ,start=None):
    #UpperLeft
    if(type==1):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=5
        endTS= 65
        startFeatures= 5 
        endFeatures = 65
        ImpFeatures=60
        DataName="UpperLeft"
        multipleBox=False

    #UpperRight
    elif(type==2):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=5
        endTS= 65
        startFeatures= 35 
        endFeatures = 95
        ImpFeatures=60
        DataName="UpperRight"
        multipleBox=False

    #UpperMiddle
    elif(type==3):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=5
        endTS= 65
        startFeatures= 20 
        endFeatures = 80
        ImpFeatures=60
        DataName="UpperMiddle"
        multipleBox=False


    #BottomLeft
    elif(type==4):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=35
        endTS= 95
        startFeatures= 5
        endFeatures = 65
        ImpFeatures=60
        DataName="BottomLeft"
        multipleBox=False

    #BottomRight
    elif(type==5):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=35
        endTS= 95
        startFeatures= 35 
        endFeatures = 95
        ImpFeatures=60
        DataName="BottomRight"
        multipleBox=False

    #BottomMiddle
    elif(type==6):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=35
        endTS= 95
        startFeatures= 20 
        endFeatures = 80
        ImpFeatures=60
        DataName="BottomMiddle"
        multipleBox=False


    #MiddleBox
    elif(type==7):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=20
        endTS= 80
        startFeatures= 20 
        endFeatures = 80
        ImpFeatures=60
        DataName="MiddleBox"
        multipleBox=False



    elif(type=="TopBox"):
        TS = 100
        NumFeatures=100
        ImpTS=30
        startTS=0
        endTS= 30
        startFeatures= 10
        endFeatures = 90
        ImpFeatures=80
        DataName="TopBox"
        multipleBox=False


    elif(type=="MiddleBox"):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=20
        endTS= 80
        startFeatures= 20 
        endFeatures = 80
        ImpFeatures=60
        DataName="MiddleBox"
        multipleBox=False



    elif(type=="ThreeUpperBoxes"):
        TS = 100
        NumFeatures=100
        ImpTS=[40,40,40]
        startTS=[0,0,0]
        endTS= [40,40,40]
        startFeatures=[0,35,70]
        endFeatures= [ 30,65, 100]
        ImpFeatures=[30,30,30]
        DataName="ThreeUpperBoxes"
        multipleBox=True


    elif(type=="ThreeMiddleBoxes"):
        TS = 100
        NumFeatures=100
        ImpTS=[40,40,40]
        startTS=[25,25,25]
        endTS= [65,65,65]
        startFeatures=[0,35,70]
        endFeatures= [ 30,65, 100]
        ImpFeatures=[30,30,30]
        DataName="ThreeMiddleBoxes"
        multipleBox=True
    elif(type=="BottomBox"):
        TS = 100
        NumFeatures=100
        ImpTS=30
        startTS=70
        endTS= 100
        startFeatures= 10
        endFeatures = 90
        ImpFeatures=80
        DataName="BottomBox"
        multipleBox=False

    elif(type=="MovingMiddleBox"):
        TS = 100
        NumFeatures=100
        ImpTS=60
        startTS=start
        endTS= startTS+60
        startFeatures= 20 
        endFeatures = 80
        ImpFeatures=60
        DataName="BottomBox"
        multipleBox=False








    referenceSample= createReferenceSample([TS,NumFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox )
    numberOfImpFeatures= getNumberOfImportantFeatures( ImpTS , ImpFeatures , multipleBox)
    return referenceSample , numberOfImpFeatures

def normalizeAtoB(score,max,min,a=0,b=1):
    newscore= (b-a)*((score-min)/(max-min))+a
    return newscore
def rescale(image,changetodiscrete=True,flat=False):

    newImage=np.zeros((image.shape))
    max_score, min_score = np.amax(image), np.amin(image)
    if(not flat):
        for i in range (image.shape[0]):
            for j in range (image.shape[1]):
                score= normalizeAtoB (image[i,j], max_score, min_score)


                if(changetodiscrete):
                    if(score>=2.5):
                        newImage[i,j]=1
                    elif(score<=-2.5):
                        newImage[i,j]=-1
                else:
                    newImage[i,j]=score
    else:
        for i in range (image.shape[0]):
            score= normalizeAtoB (image[i], max_score, min_score)
            if(changetodiscrete):
                if(score>=2.5):
                    newImage[i]=1
                elif(score<=-2.5):
                    newImage[i]=-1
            else:
                newImage[i]=score
    return newImage


def rescale_(image):
    new= minmax_scale(image)
    return new



def getIndexOfMaxValues(array,N):
    array=array.flatten()
    ind = np.argpartition(array, -1*N)[-1*N:]
    return ind

def getIndexOfImpValues(Type,start=None):
    referenceSample , numberOfImpFeatures= getBoxInfo(Type,start)
    ind = getIndexOfMaxValues(referenceSample , numberOfImpFeatures)
    return referenceSample , ind ,numberOfImpFeatures


def plotSample(sample,start=None):
    plotHeatMapExampleWise(sample,"mixedBox" ,"test",max=1,min=0,greyScale=True)



def getJaccardSimilarityScore(Orignal,Salincy):
    intersectionCount = np.intersect1d(Orignal,Salincy).shape[0]
    unionCount = np.union1d(Orignal,Salincy).shape[0]
    return intersectionCount/unionCount



def getWeightedJaccardSimilarityScore(Orignal,Salincy):

    return np.sum(np.minimum(Orignal, Salincy))/np.sum(np.maximum(Orignal, Salincy))



# def getWeightedJaccardSimilarityScore(Orignal,Salincy,OrignalIndex,SalincyIndex):
#     intersection= np.intersect1d(OrignalIndex,SalincyIndex)
#     intersectionSum=np.sum(Salincy[intersection])
#     union = np.union1d(OrignalIndex,SalincyIndex)
#     unionSum=np.sum(Orignal[union])
#     unionCount = union.shape[0]


#     return intersectionSum/unionSum


def getEnrichmentScore(Orignal,Salincy,OrignalIndex,SalincyIndex,numberOfImpFeatures):
    intersection= np.intersect1d(OrignalIndex,SalincyIndex)
    intersectionSum=np.sum(Salincy[intersection])
    union = np.union1d(OrignalIndex,SalincyIndex)
    unionSum=np.sum(Salincy[union])
    missingA = np.setdiff1d(OrignalIndex, SalincyIndex)
    missingASum=np.sum(Salincy[missingA])

    return  (intersectionSum-missingASum)/numberOfImpFeatures




