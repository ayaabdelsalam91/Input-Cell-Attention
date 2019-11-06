import torch
import Helper
import argparse
import sys
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from Helper import checkAccuracyOnTestLstm 
from net import *
from numpy import array

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Results='../Results/'

def getModelType(model, args):
    model_=""
    cellAttention=""

    if("_FF_" in model):
        model_="FF"
    elif("_LSTM_" in model):
        model_="LSTM"
    elif("_BiLSTM_" in model):
        model_="BiLSTM"
    if("_InputCellAttention_" in model):
        cellAttention="InputCellAttention"

    modelType = args.DataName
    if(model_!=""):
        modelType+="_"+model_
    if(cellAttention!=""):
        modelType +="_"+cellAttention

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
        model_output = self.model(Sample , [self.sequence_length])
        one_hot_output = torch.DoubleTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        
        model_output.backward(gradient=one_hot_output)

        grad = Sample.grad.data.cpu().numpy()
        saliency = np.absolute(grad)
        return grad , saliency


def main(args):


    Training = Helper.load_CSV(args.data_dir+"SimulatedTraining"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TrainingLabel = Helper.load_CSV(args.data_dir+"SimulatedTrainingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")


    TrainingSeqLength= TrainingLabel[:,1].astype(int)
    TrainingLabel= TrainingLabel[:,0]
    TrainingLabel=TrainingLabel.reshape(TrainingLabel.shape[0],)
    TrainingLabel = Helper.reOrderLabels(TrainingLabel.tolist())
    TrainingLabel=np.array(TrainingLabel)


    print("Loading Testing")
    Testing = Helper.load_CSV(args.data_dir+"SimulatedTestingData"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TestingLabel = Helper.load_CSV(args.data_dir+"SimulatedTestingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    
    TestingSeqLength= TestingLabel[:,1].astype(int)
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

    modelName = "Simulated"
    modelName+=args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)


    models=[
            "../Models/"+modelName+"_LSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),
            "../Models/"+modelName+"_InputCellAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),
            ]


    TestingLabel = Variable(torch.from_numpy(TestingLabel).to(device))

    stats=np.zeros((len(models),4),dtype=object)
    for m , model in enumerate(models):

        ModelType = getModelType(model,args)
        model=model+"_BEST.pkl"
        pretrained_model = torch.load(model,map_location=device) 


        VS = VanillaSaliency(pretrained_model , ModelType,args)
        grads = np.zeros((TestingRNN.shape))
        saliencies = np.zeros((TestingRNN.shape))

       
        Test_Acc  =   checkAccuracyOnTestLstm(test_loaderRNN , pretrained_model, args,Flag=True)
        print("Test Acc" , ModelType,Test_Acc )
        for i in range (TestingRNN.shape[0]):
            samples_= TestingRNN[i].reshape(1,args.sequence_length,args.input_size)

            samples = Variable(torch.from_numpy(samples_).to(device))
            samples = Variable(samples,  volatile=False, requires_grad=True)
            grad , saliency = VS.generate_gradients(samples, TestingLabel[i])
            grad = grad.reshape(args.sequence_length,args.input_size)
            grads[i]=grad
            samples_= TestingRNN[i].reshape(args.sequence_length,args.input_size)
            saliency=saliency.reshape(args.sequence_length,args.input_size)
            saliencies[i]=saliency

        grads=grads.reshape((grads.shape[0],grads.shape[1]*grads.shape[2]))
        saliencies_=saliencies.reshape((saliencies.shape[0],saliencies.shape[1]*saliencies.shape[2]))
        Helper.save_intoCSV(saliencies_,Results+ModelType+"_saliencies.csv")

    


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--input_size', type=int,default=100)
    parser.add_argument('--importance', type=str, default=0)
    parser.add_argument('--DataName', type=str ,default="TopBox")
    parser.add_argument('--attention_hops', type=int, default=20)
    parser.add_argument('--d_a', type=int, default=50)
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
