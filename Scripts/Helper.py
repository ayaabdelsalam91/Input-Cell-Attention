import sys
import os
import numpy as np
import random
import argparse
import pandas as pd
from collections import Counter
from time import time
import math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F
import itertools
from sklearn.metrics import confusion_matrix
import time
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



################################## General Helper Function ##############################

def jaccard_similarity(x,y):

    x=x.tolist()
    y=y.tolist()
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def getPixelAccuracy(true,pred):
    correct=0
    incorrect=0
    true=true.flatten()
    pred=pred.flatten()
    for i in range(true.shape[0]):
        if(true[i]==pred[i]):
            correct+=1
        else:
            incorrect+=1
    return(correct/(correct+incorrect))

def getGradient(Target,sampleSize,TargetSize,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd,multipleBox):
    sample=np.ones((sampleSize))*(-1*Target)

    if multipleBox:
        numOfBoxes=len(TargetSize[0])
        for i in range(numOfBoxes):
            Features=np.ones(([TargetSize[0][i],TargetSize[1][i]]))*Target
            sample[TargetYStart[i]:TargetYEnd[i],TargetXStart[i]:TargetXEnd[i]]=Features
    else:
        Features=np.ones((TargetSize))*Target
        sample[TargetYStart:TargetYEnd,TargetXStart:TargetXEnd]=Features
    return sample



def load_CSV(file,returnDF=False,Flip=False):
	df = pd.read_csv(file)
	data=df.values
	if(Flip):
		print("Will Un-Flip before Loading")
		data=data.reshape((data.shape[1],data.shape[0]))
	if(returnDF):
		return df
	return data

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def save_intoCSV(data,file,Flip=False,col=None,index=False):
	if(Flip):
		print("Will Flip before Saving")
		data=data.reshape((data.shape[1],data.shape[0]))
		print("New shape" , data.shape)

	df = pd.DataFrame(data)
	if(col!=None):
		df.columns = col
	df.to_csv(file,index=index)

def returnLabel(filename,labels):
	for label in labels:
		if(label in filename):
			return label
	return None

def splitSubjects(subjects,trainingPercentage):
	numberOfTrainingSubjects = math.ceil(len(subjects)*trainingPercentage)
	numberOfTestingSubjects = len(subjects) - numberOfTrainingSubjects
	trainingSubjects = random.sample(subjects, numberOfTrainingSubjects)
	testingSubjects = np.setdiff1d(subjects,trainingSubjects)
	return trainingSubjects , testingSubjects

def getPercenatgeErrorMatrix(matrix):
	dim=matrix.shape[0]
	results = np.zeros((dim,dim))
	for i in range(dim):
		for j in range (dim):
			if(i!=j):
				results[i,j]=matrix[i,j]/np.sum(matrix[i,:])*100
	return results

def changeLabelsToNumbers(Labels,Dictionary):
	outLabels=[]
	for label in Labels:
		outLabels.append(Dictionary.index(label))
	return outLabels

def reOrderLabels(Labels):
	uniqueLabels =  list(set(Labels))
	outLabels=[]
	for label in Labels:
		outLabels.append(uniqueLabels.index(label))
	return outLabels


def get_files_in_directory(a_dir):
	return [a_dir+f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]


def load_file(file):
	data = pd.read_csv(file, delimiter="\s", header=None)
	return data.values


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


