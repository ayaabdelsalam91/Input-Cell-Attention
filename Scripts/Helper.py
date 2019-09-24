import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import itertools

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



################################## General Helper Function ##############################



def load_CSV(file,returnDF=False,Flip=False):
	df = pd.read_csv(file)
	data=df.values
	if(Flip):
		print("Will Un-Flip before Loading")
		data=data.reshape((data.shape[1],data.shape[0]))
	if(returnDF):
		return df
	return data



def save_intoCSV(data,file,Flip=False,col=None,index=False):
	if(Flip):
		print("Will Flip before Saving")
		data=data.reshape((data.shape[1],data.shape[0]))


	df = pd.DataFrame(data)
	if(col!=None):
		df.columns = col
	df.to_csv(file,index=index)




def reOrderLabels(Labels):
	uniqueLabels =  list(set(Labels))
	outLabels=[]
	for label in Labels:
		outLabels.append(uniqueLabels.index(label))
	return outLabels





################################## Get Accuracy Functions ##############################



def checkAccuracyOnTestLstm(test_loader , net ,args,Flag=False , returnValue=False):
    predicted_=[]
    labels_=[]

    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for  (samples, labels,seqLength)  in test_loader:
            samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
            samples = Variable(samples)
            labels = labels.to(device)
            labels = Variable(labels).long()
            outputs = net(samples , seqLength)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_.append(predicted)
            labels_.append(labels)
       
        if(Flag):
            predicted_ = list(itertools.chain.from_iterable(predicted_))
            labels_ = list(itertools.chain.from_iterable(labels_))
    if(returnValue):
        return 100.0 * float(correct) / total , predicted_ , labels_
    else:
        return 100.0 * float(correct) / total 


