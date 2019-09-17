import numpy as np
import Helper
from heatmap import *
Loc_Graph = '../Graphs/'

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

def createDataset(NumberOFsamples,sampleSize,TargetSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd , multipleBox=False):
	DataSet = np.zeros((NumberOFsamples , sampleSize[0],sampleSize[1]))
	Targets = np.random.randint(-1, 1,NumberOFsamples)

	for i in range (NumberOFsamples):
		if(Targets[i]==0):
			Targets[i]=1
		sample = createSample(Targets[i],sampleSize,TargetSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd , multipleBox)

		DataSet[i,:,:,]=sample
	return DataSet , Targets




#TrainingDataset
#number of features 100
#number of timeSteps 10

TS = 100
NumFeatures=100
ImpTS=30
startTS=0
endTS= 30
startFeatures= 10
endFeatures = 90
ImpFeatures=80
DataName="TopBox"
DataName_="Ealier Box"
multipleBox=False


# TS = 100
# NumFeatures=100
# ImpTS=30
# startTS=70
# endTS= 100
# startFeatures= 10
# endFeatures = 90
# ImpFeatures=80
# DataName="BottomBox"
# DataName_="Latter Box"
# multipleBox=False


# TS = 100
# NumFeatures=100
# ImpTS=60
# startTS=20
# endTS= 80
# startFeatures= 20 
# endFeatures = 80
# ImpFeatures=60
# DataName="MiddleBox"
# multipleBox=False

# TS = 70
# NumFeatures=100
# ImpTS=20
# startTS=0
# endTS= 20
# startFeatures= 60 
# endFeatures = 100
# ImpFeatures=40
# DataName="UpperRight"
# multipleBox=False

# TS = 70
# NumFeatures=100
# ImpTS=20
# startTS=0
# endTS= 20
# startFeatures= 0 
# endFeatures = 40
# ImpFeatures=40
# DataName="UpperLeft"
# multipleBox=False


# TS = 100
# NumFeatures=100
# ImpTS=[40,40,40]
# startTS=[0,0,0]
# endTS= [40,40,40]
# startFeatures=[0,35,70]
# endFeatures= [ 30,65, 100]
# ImpFeatures=[30,30,30]
# DataName="ThreeUpperBoxes"
# multipleBox=True



# TS = 100
# NumFeatures=100
# ImpTS=[40,40,40]
# startTS=[25,25,25]
# endTS= [65,65,65]
# startFeatures=[0,35,70]
# endFeatures= [ 30,65, 100]
# ImpFeatures=[30,30,30]
# DataName="ThreeMiddleBoxes"
# multipleBox=True



print(DataName)
TrainingDataset  , TrainingLabel= createDataset(400,[TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox=multipleBox)
newTrainingLabel =  np.zeros((400,2))
newTrainingLabel[:,0]=TrainingLabel
newTrainingLabel[:,1]=TS
# ### Plotting####
# negIndices = [i for i, x in enumerate(TrainingLabel) if x == -1]
# posIndices = [i for i, x in enumerate(TrainingLabel) if x == 1]
# max = np.amax(TrainingDataset)  
# min = np.amin(TrainingDataset)  
# posExamplesRNN =TrainingDataset[posIndices,:,:] 
# meanPosExamples = np.mean(posExamplesRNN, axis=0)

# plotHeatMapExampleWise(meanPosExamples,"","_posMean"+DataName,max=max,min=min, flip=True,x_axis="Time",y_axis="Feature Value")
# negExamplesRNN =TrainingDataset[negIndices,:,:] 
# meanNegExamples = np.mean(negExamplesRNN, axis=0)
# plotHeatMapExampleWise(meanNegExamples,"","_negMean"+DataName,max=max, min=min  , flip=True,x_axis="Time",y_axis="Feature Value")
# ########




TestingDataset  ,  TestingLabel= createDataset(1000,[TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox=multipleBox)
newTestingLabel =  np.zeros((1000,2))
newTestingLabel[:,0]=TestingLabel
newTestingLabel[:,1]=TS

### Plotting####
# negIndices = [i for i, x in enumerate(TestingLabel) if x == -1]
# posIndices = [i for i, x in enumerate(TestingLabel) if x == 1]
# max = np.amax(TestingDataset)  
# min = np.amin(TestingDataset)  
# posExamplesRNN =TestingDataset[posIndices,:,:] 
# meanPosExamples = np.mean(posExamplesRNN, axis=0)
# plotHeatMapExampleWise(meanPosExamples,"Mean Postive " + DataName,"posMean"+DataName,max=max,min=min)
# negExamplesRNN =TestingDataset[negIndices,:,:] 
# meanNegExamples = np.mean(negExamplesRNN, axis=0)
# plotHeatMapExampleWise(meanNegExamples,"Mean Negative " + DataName,"negMean"+DataName,max=max, min=min , flip=True,x_axis="Time",y_axis="Features")
########


TrainingDataset=TrainingDataset.reshape((TrainingDataset.shape[0],TrainingDataset.shape[1]*TrainingDataset.shape[2]))
TestingDataset=TestingDataset.reshape((TestingDataset.shape[0],TestingDataset.shape[1]*TestingDataset.shape[2]))
data_dir="../Data/"
Helper.save_intoCSV(TrainingDataset,data_dir+"SimulatedTraining"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
Helper.save_intoCSV(newTrainingLabel,data_dir+"SimulatedTrainingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
Helper.save_intoCSV(TestingDataset,data_dir+"SimulatedTestingData"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
Helper.save_intoCSV(newTestingLabel,data_dir+"SimulatedTestingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")