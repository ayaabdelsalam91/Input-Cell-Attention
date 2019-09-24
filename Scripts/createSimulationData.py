import numpy as np
import Helper
import sys
import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def strTovalue(value):
	if ('[' not in value):
		return int(value)
	else:
		value= value.replace("[", "")
		value = value.replace("]", "")
		value= value.replace(" ", "")
		value = value.split(',')

		return list(map(int, value))


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


def main(args):
	TS = args.NumTimeSteps
	NumFeatures=args.NumFeatures
	ImpTS=strTovalue(args.ImpTimeSteps)
	startTS=strTovalue(args.StartImpTimeSteps)
	endTS= strTovalue(args.EndImpTimeSteps)
	startFeatures= strTovalue(args.StartImpFeatures)
	endFeatures = strTovalue(args.EndImpFeatures)
	ImpFeatures=strTovalue(args.ImpFeatures)
	DataName=args.DataName
	multipleBox=args.multipleBox



	print("Creating Training Dataset")

	TrainingDataset  , TrainingLabel= createDataset(args.NumTrainingSamples,[TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox=multipleBox)
	newTrainingLabel =  np.zeros((args.NumTrainingSamples,2))
	newTrainingLabel[:,0]=TrainingLabel
	newTrainingLabel[:,1]=TS



	print("Creating Testing Dataset")
	TestingDataset ,TestingLabel= createDataset(args.NumTestingSamples,[TS,NumFeatures],[ImpTS,ImpFeatures],startTS,endTS,startFeatures,endFeatures,multipleBox=multipleBox)
	newTestingLabel =  np.zeros((args.NumTestingSamples,2))
	newTestingLabel[:,0]=TestingLabel
	newTestingLabel[:,1]=TS

	print("Saving Datasets...")
	TrainingDataset=TrainingDataset.reshape((TrainingDataset.shape[0],TrainingDataset.shape[1]*TrainingDataset.shape[2]))
	TestingDataset=TestingDataset.reshape((TestingDataset.shape[0],TestingDataset.shape[1]*TestingDataset.shape[2]))
	data_dir="../Data/"
	Helper.save_intoCSV(TrainingDataset,data_dir+"SimulatedTraining"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	Helper.save_intoCSV(newTrainingLabel,data_dir+"SimulatedTrainingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	Helper.save_intoCSV(TestingDataset,data_dir+"SimulatedTestingData"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")
	Helper.save_intoCSV(newTestingLabel,data_dir+"SimulatedTestingLabels"+DataName+"_F"+str(NumFeatures)+"_TS_"+str(TS)+"_IMPSTART_"+str(startTS)+".csv")

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--DataName', type=str, default="TopBox")
	parser.add_argument('--NumTrainingSamples',type=int,default=1000)
	parser.add_argument('--NumTestingSamples',type=int,default=300)
	parser.add_argument('--NumTimeSteps',type=int,default=100)
	parser.add_argument('--NumFeatures',type=int,default=100)
	parser.add_argument('--ImpTimeSteps',type=str,default="30")
	parser.add_argument('--ImpFeatures',type=str,default="80")
	parser.add_argument('--StartImpTimeSteps',type=str,default="0")
	parser.add_argument('--EndImpTimeSteps',type=str,default="30")
	parser.add_argument('--StartImpFeatures',type=str,default="10")
	parser.add_argument('--EndImpFeatures',type=str,default="90")
	parser.add_argument('--multipleBox',type=str2bool ,default=False)
	return  parser.parse_args()

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))

