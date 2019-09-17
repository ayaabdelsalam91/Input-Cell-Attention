import numpy as np
import Helper
from heatmap import *

TS = 100
NumFeatures=100

TypesArray=["UpperLeft" , "UpperRight" , "UpperMiddle" , "BottomLeft" ,"BottomRight" , "BottomMiddle" , "MiddleBox"]

def getBoxType(type):
	#UpperLeft
	if(type==1):
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
		ImpTS=60
		startTS=20
		endTS= 80
		startFeatures= 20 
		endFeatures = 80
		ImpFeatures=60
		DataName="MiddleBox"
		multipleBox=False




	return [TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox 


def createSample(Target,sampleSize,TargetSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd,multipleBox):
	sample=np.random.normal(0,1,sampleSize)

	if multipleBox:
		numOfBoxes=len(TargetSize[0])
		for i in range(numOfBoxes):
			Features=np.random.normal(Target,1,[TargetSize[0][i],TargetSize[1][i]])
			sample[TargetYStart[i]:TargetYEnd[i],TargetXStart[i]:TargetXEnd[i]]=Features
	else:
		Features=np.random.normal(Target,1,TargetSize)
		sample[TargetYStart:TargetYEnd,TargetXStart:TargetXEnd]=Features
	return sample

def createDataset(NumberOFsamples):
	DataSet = np.zeros((NumberOFsamples , TS,NumFeatures))
	Targets = np.random.randint(-1, 1,NumberOFsamples)
	Type = np.random.randint(1, 8,NumberOFsamples)
	for i in range (NumberOFsamples):
		if(Targets[i]==0):
			Targets[i]=1
		sampleSize,TargetSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd , multipleBox = getBoxType(Type[i])
		sample = createSample(Targets[i],sampleSize,TargetSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd , multipleBox)

		DataSet[i,:,:,]=sample
	return DataSet , Targets , Type




#TrainingDataset
#number of features 100
#number of timeSteps 10



DataName="MixedBoxes"
print(DataName,8000)
TrainingDataset  , TrainingLabel,Type= createDataset(8000)
newTrainingLabel =  np.zeros((8000,3))
newTrainingLabel[:,0]=TrainingLabel
newTrainingLabel[:,1]=TS
newTrainingLabel[:,2]=Type

### Plotting####
# max=np.max(TrainingDataset)
# min=np.min(TrainingDataset)
# print(max,min)
# for i in range(1,8):
# 	for j in range(2000):
# 		# print("Here" , newTrainingLabel[j,2],TypesArray[i])
# 		if(newTrainingLabel[j,2]==i):

# 			plotHeatMapExampleWise(TrainingDataset[j], TypesArray[i-1]+" Example" ,TypesArray[i-1]+"Example",max=max, min=min , greyScale=True)
# 			break
########



TestingDataset  ,  TestingLabel,Type= createDataset(1000)
newTestingLabel =  np.zeros((1000,3))
newTestingLabel[:,0]=TestingLabel
newTestingLabel[:,1]=TS
newTestingLabel[:,2]=Type

# ## Plotting####
# max=np.max(TestingDataset)
# min=np.min(TestingDataset)
# print(max,min)
# for i in range(1,8):
# 	for j in range(1000):
# 		# print("Here" , newTrainingLabel[j,2],TypesArray[i])
# 		if(newTestingLabel[j,2]==i):

# 			plotHeatMapExampleWise(newTrainingLabel[j], TypesArray[i-1]+" Example" ,TypesArray[i-1]+"Example",max=max, min=min , greyScale=True)
# 			break
# #######


TrainingDataset=TrainingDataset.reshape((TrainingDataset.shape[0],TrainingDataset.shape[1]*TrainingDataset.shape[2]))
TestingDataset=TestingDataset.reshape((TestingDataset.shape[0],TestingDataset.shape[1]*TestingDataset.shape[2]))
data_dir="../Data/"
Helper.save_intoCSV(TrainingDataset,data_dir+"SimulatedTraining"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+".csv")
Helper.save_intoCSV(newTrainingLabel,data_dir+"SimulatedTrainingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+".csv")


Helper.save_intoCSV(TestingDataset,data_dir+"SimulatedTestingData"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+".csv")
Helper.save_intoCSV(newTestingLabel,data_dir+"SimulatedTestingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+".csv")