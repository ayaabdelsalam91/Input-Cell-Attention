"""
Created on Thu Oct 26 11:19:58 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from net import *
from misc_functions import convert_to_grayscale, save_gradient_images,get_sample
import Helper
import argparse
import random
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import Helper
import torch.utils.data as data_utils
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from Helper import checkAccuracyOnTestLstm , checkAccuracyOnTestLstmCell ,checkAccuracyOnTestFF
from net import *
from numpy import array
from numpy.linalg import norm
from heatmap import *
import random
from customLSTM import *
from sklearn.metrics import jaccard_similarity_score
from scipy.spatial import distance
np.set_printoptions(threshold=sys.maxsize)
from scipy.spatial.distance  import jaccard
from accuracyMethods import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Loc_Graph = '../Graphs/'
Results='../Results/'

typeName = ["UpperLeft","UpperRight","UpperMiddle","BottomLeft","BottomRight","BottomMiddle","MiddleBox"]

def getModelType(model, args):
    model_=""
    cellAttention=""
    pooling=""
    selfAttention=""
    if("_FF_" in model):
        model_="FF"
    elif("_UniLSTM_" in model):
        model_="UniLSTM"
    elif("_BiLSTM_" in model):
        model_="BiLSTM"
    if("_InputCellAttention_" in model):
        cellAttention="InputCellAttention"
    elif("_InputCellAttentionVector_" in model):
        cellAttention="InputCellAttentionVector"
    elif("_InputCellAttentionMatrix_" in model):
        cellAttention="InputCellAttentionMatrix"
    elif("_InputCellAttentionMat" in model):
        cellAttention="InputCellAttentionMat"
    elif("_XtAndInputCellAttention_" in model):
        cellAttention="XtAndInputCellAttention"
    if("_MAXPOOLING_" in model):
        pooling="MAXPOOLING"
    elif("_MEANPOOLING_" in model):
        pooling="MEANPOOLING"

    if("_SelfAttention_" in model):
        selfAttention="SelfAttention"


    modelType = args.DataName
    if(model_!=""):
        modelType+="_"+model_
    if(cellAttention!=""):
        modelType +="_"+cellAttention
    if(pooling!=""):
        modelType +="_"+pooling
    if(selfAttention!=""):
        modelType+="_"+selfAttention


    return modelType

class VanillaSaliency():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model,type,args):
        self.type =  type
        self.model = model
        self.sequence_length=args.sequence_length

    def generate_gradients(self, Sample, target_class):
        # Forward
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        if("FF" in self.type):
            model_output = self.model(Sample)
        else:
            model_output = self.model(Sample , [self.sequence_length])
        one_hot_output = torch.DoubleTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        
        model_output.backward(gradient=one_hot_output)

        grad = Sample.grad.data.cpu().numpy()
        # print(grad.shape)
        # # grad=grad.reshape(grad.shape[0],grad.shape[-1])
        # print(grad.shape)
        saliency = np.absolute(grad)
        return grad , saliency


def main(args):


    
    # Testing = Helper.load_CSV(args.data_dir+"SimulatedTestingData"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    # TestingLabel = Helper.load_CSV(args.data_dir+"SimulatedTestingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    
    # TestingSeqLength= TestingLabel[:,1].astype(int)
    # TestingType=TestingLabel[:,-1]
    # Types = np.unique(TestingType)
    # TestingLabel= TestingLabel[:,0]
    # TestingLabel=TestingLabel.reshape(TestingLabel.shape[0],)
    # # negIndices = [i for i, x in enumerate(TestingLabel) if x == -1]
    # # posIndices = [i for i, x in enumerate(TestingLabel) if x == 1]

    # TestingLabel = Helper.reOrderLabels(TestingLabel.tolist())
    # TestingLabel=np.array(TestingLabel)

    # scaler = StandardScaler()
    # scaler.fit(Testing)
    # Testing = scaler.transform(Testing)

    # TestingRNN = Testing.reshape(Testing.shape[0] , args.sequence_length,args.input_size)

    # test_data = data_utils.TensorDataset(torch.from_numpy(Testing),torch.from_numpy( TestingLabel))
    # test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # test_dataRNN = data_utils.TensorDataset(torch.from_numpy(TestingRNN),torch.from_numpy( TestingLabel) , torch.from_numpy(TestingSeqLength))
    # test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)

    Training = Helper.load_CSV(args.data_dir+"SimulatedTraining"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TrainingLabel = Helper.load_CSV(args.data_dir+"SimulatedTrainingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")


    TrainingSeqLength= TrainingLabel[:,1].astype(int)
    TrainingType=TrainingLabel[:,-1]
    Types = np.unique(TrainingType)
    TrainingLabel= TrainingLabel[:,0]
    TrainingLabel=TrainingLabel.reshape(TrainingLabel.shape[0],)
    TrainingLabel = Helper.reOrderLabels(TrainingLabel.tolist())
    TrainingLabel=np.array(TrainingLabel)


    print("Loading Testing")
    Testing = Helper.load_CSV(args.data_dir+"SimulatedTestingData"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TestingLabel = Helper.load_CSV(args.data_dir+"SimulatedTestingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    
    TestingSeqLength= TestingLabel[:,1].astype(int)
    TestingType=TestingLabel[:,-1]
    Types = np.unique(TestingType)
    TestingLabel= TestingLabel[:,0]
    TestingLabel=TestingLabel.reshape(TestingLabel.shape[0],)
    TestingLabel = Helper.reOrderLabels(TestingLabel.tolist())
    TestingLabel=np.array(TestingLabel)

    scaler = StandardScaler()
    scaler.fit(Training)
    Training = scaler.transform(Training)
    Testing = scaler.transform(Testing)

    TrainingRNN = Training.reshape(Training.shape[0] , args.sequence_length,args.input_size)
    TestingRNN = Testing.reshape(Testing.shape[0] , args.sequence_length,args.input_size)
  

    train_dataRNN = data_utils.TensorDataset(torch.from_numpy(TrainingRNN), torch.from_numpy(TrainingLabel) , torch.from_numpy(TrainingSeqLength))
    train_loaderRNN = data_utils.DataLoader(train_dataRNN, batch_size=args.batch_size, shuffle=True)


    test_dataRNN = data_utils.TensorDataset(torch.from_numpy(TestingRNN),torch.from_numpy( TestingLabel) , torch.from_numpy(TestingSeqLength))
    test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)


    tran_data = data_utils.TensorDataset(torch.from_numpy(Training), torch.from_numpy(TrainingLabel))
    train_loader = data_utils.DataLoader(tran_data, batch_size=args.batch_size, shuffle=True)


    test_data = data_utils.TensorDataset(torch.from_numpy(Testing),torch.from_numpy( TestingLabel))
    test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
  
    print(args.DataName)
    modelName = "Simulated"
    modelName+=args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)



    models=[
            # "../Models/"+modelName+"_FF_L"+ str(args.num_layers)+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_dropOut"+str(args.dropout)+'_'+str(args.num_classes),
            
            # "../Models/"+modelName+"_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            "../Models/"+modelName+"_InputCellAttentionMatrix_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),
            "../Models/"+modelName+"_InputCellAttentionVector_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),

            # # "../Models/"+modelName+"_XtAndInputCellAttention_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),

            # "../Models/"+modelName+"_BiLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            
            # "../Models/"+modelName+"_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            # "../Models/"+modelName+"_InputCellAttention_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            # # "../Models/"+modelName+"_XtAndInputCellAttention_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),

            # "../Models/"+modelName+"_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            # "../Models/"+modelName+"_InputCellAttention_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            # # "../Models/"+modelName+"_XtAndInputCellAttention_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            
            "../Models/"+modelName+"_UniLSTM_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
            # "../Models/"+modelName+"_InputCellAttention_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),
            # "../Models/"+modelName+"_XtAndInputCellAttention_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
            ]


    for m , model in enumerate(models):

        ModelType = getModelType(model,args)
        print(ModelType)
        model=model+"_BEST.pkl"
        pretrained_model = torch.load(model,map_location='cpu') 


        VS = VanillaSaliency(pretrained_model , ModelType,args)
        grads = np.zeros((TestingRNN.shape))
        saliencies = np.zeros((TestingRNN.shape))
        JaccardScore=0
        Test_Acc=0
        euclideanScore=0
        count=0
        stats[m,0]=ModelType
        if( not("FF" in ModelType)):
            Test_Acc  =   checkAccuracyOnTestLstm(test_loaderRNN , pretrained_model, args,Flag=True)
            # Train_Acc  =   checkAccuracyOnTestLstm(train_loaderRNN , pretrained_model, args,Flag=True)
            print("Test Acc" , ModelType,Test_Acc )
        for i in range (TestingRNN.shape[0]):

            if("FF" in ModelType):
  
                    samples_= TestingRNN[i].reshape(1,int(args.input_size*args.sequence_length))
            else:
        
                    samples_= TestingRNN[i].reshape(1,args.sequence_length,args.input_size)

            samples = Variable(torch.from_numpy(samples_))
            samples = Variable(samples,  volatile=False, requires_grad=True)
            grad , saliency = VS.generate_gradients(samples, TestingLabel[i])
            grad = grad.reshape(args.sequence_length,args.input_size)
            grads[i]=grad
            samples_= TestingRNN[i].reshape(args.sequence_length,args.input_size)
            saliency=saliency.reshape(args.sequence_length,args.input_size)
            saliencies[i]=saliency

        grads=grads.reshape((grads.shape[0],grads.shape[1]*grads.shape[2]))
        saliencies_=saliencies.reshape((saliencies.shape[0],saliencies.shape[1]*saliencies.shape[2]))
        Helper.save_intoCSV(saliencies_,Results+args.DataName+ModelType+str(args.importance)+"saliencies.csv")

        # # ############### For Plotting    ################  

        # for j in range(Types.shape[0]):
        #     posGrad=[]
        #     negGrad=[]
        #     posSample=[]
        #     negSample=[]
        #     for i in range (TestingRNN.shape[0]):
        #         # if(TestingType[i]==Types[j]):
        #             if(TestingLabel[i]==0):
        #                 negGrad.append(saliencies[i])
        #                 negSample.append(TestingRNN[i])
        #             else:
        #                 posGrad.append(saliencies[i])
        #                 posSample.append(TestingRNN[i])

        #     meanPosGrad = np.mean(np.array(posGrad), axis=0)
        #     meanNegGrad = np.mean(np.array(negGrad), axis=0)
        #     meanPosSample = np.mean(np.array(posSample), axis=0)
        #     meanNegSample = np.mean(np.array(negSample), axis=0)

        #     # max = np.amax(meanPosGrad)  
        #     # min = np.amin(meanPosGrad)  
        #     # print(meanPosGrad.shape)
        #     # plotHeatMapExampleWise(meanPosGrad,ModelType+" Postive Grad "+typeName[j] ,ModelType+"posGradMeanMat"+typeName[j],max=max,min=min,greyScale=True,flip=True)
        #     # linePlot(meanPosGrad ,ModelType+" Postive Grad line graph "+typeName[j],ModelType+"linePosMeanMat"+typeName[j])

        #     # max = np.amax(meanNegGrad)  
        #     # min = np.amin(meanNegGrad) 
        #     # plotHeatMapExampleWise(meanNegGrad,ModelType+" Negative Grad "+typeName[j] ,ModelType+"negGradMeanMat"+typeName[j],max=max,min=min,greyScale=True,flip=True)
        #     # linePlot(meanNegGrad ,ModelType+" Negative Grad line graph "+typeName[j] ,ModelType+"lineNegMeanMat"+typeName[j] )

        #     # max = np.amax(TestingRNN)  
        #     # min = np.amin(TestingRNN)  

        #     # # meanPosExamples = np.mean(posExamplesRNN, axis=0)
        #     # print(meanPosSample.shape)
        #     # plotHeatMapExampleWise(meanPosSample,"","_posMean",max=max,min=min, flip=True,x_axis="Time",y_axis="Feature Value")
        #     # plotHeatMapExampleWise(meanNegSample,"","_negMean",max=max, min=min  , flip=True,x_axis="Time",y_axis="Feature Value")

        #     max = np.amax(meanPosGrad)  
        #     min = np.amin(meanPosGrad) 
        #     plotHeatMapExampleWise(meanPosGrad,ModelType+" Postive Grad "+typeName[j] ,ModelType+"PosSample"+typeName[j],max=max,min=min,flip=True,greyScale=True)
        #     # # linePlot(meanPosSample ,ModelType+" Postive Grad line graph "+typeName[j],ModelType+"linePosMeanMatmeanPosSample"+typeName[j])

        #     max = np.amax(meanNegGrad)  
        #     min = np.amin(meanNegGrad) 
        #     plotHeatMapExampleWise(meanNegGrad,ModelType+" Negative Grad "+typeName[j] ,ModelType+"negSample"+typeName[j],max=max,min=min,flip=True , greyScale=True)
        #     # # linePlot(meanNegGrad ,ModelType+" Negative Grad line graph "+typeName[j] ,ModelType+"lineNegMeanMat"+typeName[j] )




    os.system('say "your program has finished"')


  
def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--importance', type=str, default=0)
    parser.add_argument('--DataName', type=str ,default="MovingMiddleBox")
    parser.add_argument('--attention_hops', type=int, default=20)
    parser.add_argument('--d_a', type=int, default=50)
    parser.add_argument('--input_size', type=int,default=100)
    parser.add_argument('--hidden_size1', type=int,default=5)
    parser.add_argument('--num_layers', type=int,default=1)
    parser.add_argument('--batch_size', type=int,default=150)
    parser.add_argument('--num_epochs', type=int,default=100)
    parser.add_argument('--learning_rate', type=float,default=0.001)
    parser.add_argument('--debug', type=bool,default=False)
    parser.add_argument('--rnndropout', type=float,default=0.1)
    parser.add_argument('--LSTMdropout', type=float,default=0.2)
    parser.add_argument('--dropout', type=float,default=0.2)
    parser.add_argument('--data-dir', help='Data  directory', action='store', type=str ,default="../Data/")
    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))