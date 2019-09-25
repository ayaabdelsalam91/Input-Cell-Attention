import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import Helper
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from Helper import checkAccuracyOnTestLstm 
from net import *
from numpy import array


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True
print(device)





def main(args):
    criterion = nn.CrossEntropyLoss()

    # print("Loading Training")
    Training = Helper.load_CSV(args.data_dir+"SimulatedTraining"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TrainingLabel = Helper.load_CSV(args.data_dir+"SimulatedTrainingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")

    TrainingSeqLength= TrainingLabel[:,-1].astype(int)

    TrainingLabel= TrainingLabel[:,0]
    TrainingLabel=TrainingLabel.reshape(TrainingLabel.shape[0],)

    TrainingLabel = Helper.reOrderLabels(TrainingLabel.tolist())
    TrainingLabel=np.array(TrainingLabel)


    # print("Loading Testing")
    Testing = Helper.load_CSV(args.data_dir+"SimulatedTestingData"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TestingLabel = Helper.load_CSV(args.data_dir+"SimulatedTestingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    
    TestingSeqLength= TestingLabel[:,-1].astype(int)
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
  
    print(args.DataName)

    train_dataRNN = data_utils.TensorDataset(torch.from_numpy(TrainingRNN), torch.from_numpy(TrainingLabel) , torch.from_numpy(TrainingSeqLength))
    train_loaderRNN = data_utils.DataLoader(train_dataRNN, batch_size=args.batch_size, shuffle=True)


    test_dataRNN = data_utils.TensorDataset(torch.from_numpy(TestingRNN),torch.from_numpy( TestingLabel) , torch.from_numpy(TestingSeqLength))
    test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)


    tran_data = data_utils.TensorDataset(torch.from_numpy(Training), torch.from_numpy(TrainingLabel))
    train_loader = data_utils.DataLoader(tran_data, batch_size=args.batch_size, shuffle=True)


    test_data = data_utils.TensorDataset(torch.from_numpy(Testing),torch.from_numpy( TestingLabel))
    test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    modelName="Simulated"
    modelName+=args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)




    ########################################### LSTM ##########################################
    print("Start Training" , modelName ,'LSTM')


    netUniLstmCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout ,networkType="LSTM" ).to(device)
    netUniLstmCellAtten.double()
    optimizerTimeAtten = torch.optim.Adam(netUniLstmCellAtten.parameters(), lr=args.learning_rate)


    saveModelName="../Models/"+modelName+"_LSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
    saveModelBestName =saveModelName +"_BEST.pkl"
    saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # Train the model
    total_step = len(train_loaderRNN)
    Train_acc_flag=False
    Train_Acc=0
    Test_Acc=0
    BestAcc=0
    BestEpochs = 0
    for epoch in range(args.num_epochs):
        for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
            netUniLstmCellAtten.train()
            samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
            samples = Variable(samples)
            labels = labels.to(device)
            labels = Variable(labels).long()

            outputs = netUniLstmCellAtten(samples,seqLength)
            loss = criterion(outputs, labels)
            
            optimizerTimeAtten.zero_grad()
            loss.backward()
            optimizerTimeAtten.step()

            if (i+1) % 10 == 0:
                Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmCellAtten,args)
                Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmCellAtten,args)
                if(Test_Acc>BestAcc):
                    BestAcc=Test_Acc
                    BestEpochs = epoch+1
                    torch.save(netUniLstmCellAtten, saveModelBestName)
                print ('LSTM for {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
                       .format(args.DataName, epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


            if(epoch+1)%10==0:
                torch.save(netUniLstmCellAtten, saveModelLastName)
            if(Train_Acc==100):
                torch.save(netUniLstmCellAtten,saveModelLastName)
                Train_acc_flag=True
                break
        if(Train_acc_flag):
            break

    print(">>>>>>>>>>>>> 1 layer LSTM  >>>>>>>>>>>>>>>>>>>>>>")

    Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmCellAtten, args,Flag=True)
    print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))



    ########################################### InputCellAttention ##########################################
    print("Start Training" , modelName ,'Input Cell Attention')


    netUniLstmCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout,d_a=args.d_a , r=args.attention_hops ,networkType="InputCellAttention" ).to(device)
    netUniLstmCellAtten.double()
    optimizerTimeAtten = torch.optim.Adam(netUniLstmCellAtten.parameters(), lr=args.learning_rate)


    saveModelName="../Models/"+modelName+"_InputCellAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
    saveModelBestName =saveModelName +"_BEST.pkl"
    saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # Train the model
    total_step = len(train_loaderRNN)
    Train_acc_flag=False
    Train_Acc=0
    Test_Acc=0
    BestAcc=0
    BestEpochs = 0
    for epoch in range(args.num_epochs):
        for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
            netUniLstmCellAtten.train()
            samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
            samples = Variable(samples)
            labels = labels.to(device)
            labels = Variable(labels).long()

            outputs = netUniLstmCellAtten(samples,seqLength)
            loss = criterion(outputs, labels)
            
            optimizerTimeAtten.zero_grad()
            loss.backward()
            optimizerTimeAtten.step()
            if (i+1) % 3 == 0:
                Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmCellAtten,args)
                Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmCellAtten,args)
                if(Test_Acc>BestAcc):
                    BestAcc=Test_Acc
                    BestEpochs = epoch+1
                    torch.save(netUniLstmCellAtten, saveModelBestName)
                print ('Input Cell Attention for LSTM {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
                       .format(args.DataName, epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


            if(epoch+1)%10==0:
                torch.save(netUniLstmCellAtten, saveModelLastName)
            if(Train_Acc==100):
                torch.save(netUniLstmCellAtten,saveModelLastName)
                Train_acc_flag=True
                break
        if(Train_acc_flag):
            break

    print(">>>>>>>>>>>>> 1 layer Input Cell Attention  >>>>>>>>>>>>>>>>>>>>>>")

    Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmCellAtten, args,Flag=True)
    print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--importance', type=str, default=0)
    parser.add_argument('--DataName', type=str ,default="TopBox")
    parser.add_argument('--attention_hops', type=int, default=20)
    parser.add_argument('--d_a', type=int, default=50)
    parser.add_argument('--input_size', type=int,default=100)
    parser.add_argument('--hidden_size1', type=int,default=5)
    parser.add_argument('--num_layers', type=int,default=1)
    parser.add_argument('--batch_size', type=int,default=50)
    parser.add_argument('--num_epochs', type=int,default=400)
    parser.add_argument('--learning_rate', type=float,default=0.001)
    parser.add_argument('--debug', type=bool,default=False)
    parser.add_argument('--rnndropout', type=float,default=0.1)
    parser.add_argument('--LSTMdropout', type=float,default=0.2)
    parser.add_argument('--dropout', type=float,default=0.2)
    parser.add_argument('--data-dir', help='Data  directory', action='store', type=str ,default="../Data/")
    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))