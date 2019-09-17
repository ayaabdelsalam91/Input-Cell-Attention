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






TS = 100
NumFeatures=100
ImpTS=60
startFeatures= 20 
endFeatures = 80
ImpFeatures=60
DataName="MovingMiddleBox"
multipleBox=False
startTS=0
endTS=startTS+60
posMeans=[]
posMax=[]
posMin=[]

negMeans=[]
negMax=[]
negMin=[]


while (endTS<=100):
	print(startTS,endTS )

	TrainingDataset  , TrainingLabel= createDataset(1000,[TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox=multipleBox)
	newTrainingLabel =  np.zeros((1000,2))
	newTrainingLabel[:,0]=TrainingLabel
	newTrainingLabel[:,1]=TS

	# ### Plotting####
	# negIndices = [i for i, x in enumerate(TrainingLabel) if x == -1]
	# posIndices = [i for i, x in enumerate(TrainingLabel) if x == 1]
	# max = np.amax(TrainingDataset)  
	# min = np.amin(TrainingDataset)  
	# posExamplesRNN =TrainingDataset[posIndices,:,:] 
	# meanPosExamples = np.mean(posExamplesRNN, axis=0)
	# posMeans.append(meanPosExamples)
	# # plotHeatMapExampleWise(meanPosExamples,"Mean Postive " + DataName,"posMean"+DataName,max=max,min=min,greyScale=True,flip=True)
	# # plotHeatMapExampleWise(meanPosExamples,"","posMean"+DataName,max=max,min=min,greyScale=True,flip=True)

	# negExamplesRNN =TrainingDataset[negIndices,:,:] 
	# negMax.append(np.max(negExamplesRNN))
	# negMin.append(np.min(negExamplesRNN))
	# meanNegExamples = np.mean(negExamplesRNN, axis=0)
	# negMeans.append(meanNegExamples)
	# # plotHeatMapExampleWise(meanNegExamples,"Mean Negative " + DataName,"negMean"+DataName,max=max, min=min,greyScale=True,flip=True)
	# ########


	TestingDataset  ,  TestingLabel= createDataset(200,[TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox=multipleBox)
	newTestingLabel =  np.zeros((200,2))
	newTestingLabel[:,0]=TestingLabel
	newTestingLabel[:,1]=TS

	# ### Plotting####
	# negIndices = [i for i, x in enumerate(TestingLabel) if x == -1]
	# posIndices = [i for i, x in enumerate(TestingLabel) if x == 1]
	# max = np.amax(TestingDataset)  
	# min = np.amin(TestingDataset)  
	# posExamplesRNN =TestingDataset[posIndices,:,:] 
	# meanPosExamples = np.mean(posExamplesRNN, axis=0)
	# plotHeatMapExampleWise(meanPosExamples,"Mean Postive " + DataName,"posMean"+DataName,max=max,min=min)
	# negExamplesRNN =TestingDataset[negIndices,:,:] 
	# meanNegExamples = np.mean(negExamplesRNN, axis=0)
	# plotHeatMapExampleWise(meanNegExamples,"Mean Negative " + DataName,"negMean"+DataName,max=max, min=min)
	# ########


	TrainingDataset=TrainingDataset.reshape((TrainingDataset.shape[0],TrainingDataset.shape[1]*TrainingDataset.shape[2]))
	TestingDataset=TestingDataset.reshape((TestingDataset.shape[0],TestingDataset.shape[1]*TestingDataset.shape[2]))
	data_dir="../Data/"
	Helper.save_intoCSV(TrainingDataset,data_dir+"SimulatedTraining"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	Helper.save_intoCSV(newTrainingLabel,data_dir+"SimulatedTrainingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	Helper.save_intoCSV(TestingDataset,data_dir+"SimulatedTestingData"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	Helper.save_intoCSV(newTestingLabel,data_dir+"SimulatedTestingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	startTS+=10
	endTS=startTS+60


fig,ax  = plt.subplots(1, 5,sharex=True, figsize=(8,2))
fig.suptitle('Moving Box Data Sample')

for i in range(len(negMeans)):
	input=np.transpose(negMeans[i])
	ax[i].imshow(input, interpolation='nearest', cmap='seismic' , vmax=negMax[i],vmin=negMin[i])
	# ax[i].set_xlim(0,100)
	# ax[i].set_ylim(0,100)
	ax[i].axis('off')


# fig.text(0.5, 0.04, 'Time', ha='center')
# fig.text(0.1, 0.5, 'Features', va='center', rotation='vertical')
# plt.axis('off')
# plt.savefig(Loc_Graph+"MovingMiddleBox"+'.png' , box_inches='tight' ,  pad_inches = 0)
# plt.show()
