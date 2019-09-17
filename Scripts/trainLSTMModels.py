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
from Helper import checkAccuracyOnTestLstm , checkAccuracyOnTestLstmCell ,checkAccuracyOnTestFF,checkAccuracyOnTestAttentionLstm
from net import *
from numpy import array
from numpy.linalg import norm
# from heatmap import *
import random
import os


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



    # ############################################ FF network #####################################################

    # netFFL1 = FFL1(args.input_size*args.sequence_length, args.hidden_size1,args.dropout ,args.num_classes).to(device)
    # netFFL1.double()
    # optimizerFFL1 = torch.optim.Adam(netFFL1.parameters(), lr=args.learning_rate)


    # print("Start Training" , modelName , 'FF')
    # saveModelName="../Models/"+modelName+"_FF_L"+ str(args.num_layers)+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_dropOut"+str(args.dropout)+'_'+str(args.num_classes)

    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
   

    # # Train the model
    # total_step = len(train_loader)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0

    # pytorch_total_params = sum(p.numel() for p in netFFL1.parameters() if p.requires_grad)
    # orginalInput_size =  args.input_size
    # args.input_size =  args.input_size*args.sequence_length

    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
    #         netFFL1.train()
    #         samples = Variable(samples.view(-1, args.input_size)).to(device) 
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

           
    #         outputs = netFFL1(samples)    # Forward pass: compute the output class given a sampl
    #         loss = criterion(outputs, labels)
            
    #         # Backward and optimize
    #         optimizerFFL1.zero_grad()
    #         loss.backward()
    #         optimizerFFL1.step()


    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestFF(test_loader , netFFL1,args)
    #             Train_Acc = checkAccuracyOnTestFF(train_loader , netFFL1,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netFFL1, saveModelBestName)
    #             print ('FF for  {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netFFL1, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netFFL1,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer  FeedForward >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestFF(train_loader , netFFL1, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))
    # args.input_size = orginalInput_size


    # ########################################### Training LSTM ##########################################
    # print("Start Training" , modelName ,'LSTM')

    # netUniLstm = UniLSTM(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout).to(device)
    # netUniLstm.double()
    # optimizerUniLstm = torch.optim.Adam(netUniLstm.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_UniLSTMTEMP_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstm.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstm(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstm.zero_grad()
    #         loss.backward()
    #         optimizerUniLstm.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstm,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstm,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstm, saveModelBestName)
    #             print ('LSTM for {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstm, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstm,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer Unidirectional LSTM >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstm, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))



    ########################################### InputCellAttention LSTM ##########################################
    print("Start Training" , modelName ,'InputCellAttention')


    netUniLstmCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout,d_a=args.d_a , r=args.attention_hops ,networkType="InputCellAttentionMatrix" ).to(device)
    netUniLstmCellAtten.double()
    optimizerTimeAtten = torch.optim.Adam(netUniLstmCellAtten.parameters(), lr=args.learning_rate)


    saveModelName="../Models/"+modelName+"_InputCellAttentionMatrix_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
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
                print ('LSTM InputCellAttention for {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
                       .format(args.DataName, epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


            if(epoch+1)%10==0:
                torch.save(netUniLstmCellAtten, saveModelLastName)
            if(Train_Acc==100):
                torch.save(netUniLstmCellAtten,saveModelLastName)
                Train_acc_flag=True
                break
        if(Train_acc_flag):
            break

    print(">>>>>>>>>>>>> 1 layer InputCellAttention Matrix LSTM >>>>>>>>>>>>>>>>>>>>>>")

    Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmCellAtten, args,Flag=True)
    print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))


    ########################################### InputCellAttention LSTM ##########################################
    print("Start Training" , modelName ,'InputCellAttention')


    netUniLstmCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout,d_a=args.d_a , r=args.attention_hops ,networkType="InputCellAttentionVector" ).to(device)
    netUniLstmCellAtten.double()
    optimizerTimeAtten = torch.optim.Adam(netUniLstmCellAtten.parameters(), lr=args.learning_rate)


    saveModelName="../Models/"+modelName+"_InputCellAttentionVector_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
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
                print ('LSTM InputCellAttention for {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
                       .format(args.DataName, epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


            if(epoch+1)%10==0:
                torch.save(netUniLstmCellAtten, saveModelLastName)
            if(Train_Acc==100):
                torch.save(netUniLstmCellAtten,saveModelLastName)
                Train_acc_flag=True
                break
        if(Train_acc_flag):
            break

    print(">>>>>>>>>>>>> 1 layer InputCellAttention Vector LSTM >>>>>>>>>>>>>>>>>>>>>>")

    Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmCellAtten, args,Flag=True)
    print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))





    # ########################################### XtAndInputCellAttention LSTM ##########################################
    # print("Start Training" , modelName ,'XtAndInputCellAttention')


    # netUniLstmCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout,d_a=args.d_a , r=args.attention_hops ,networkType="XtAndInputCellAttention" ).to(device)
    # netUniLstmCellAtten.double()
    # optimizerTimeAtten = torch.optim.Adam(netUniLstmCellAtten.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_XtAndInputCellAttention_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmCellAtten.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmCellAtten(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerTimeAtten.zero_grad()
    #         loss.backward()
    #         optimizerTimeAtten.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmCellAtten,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmCellAtten,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmCellAtten, saveModelBestName)
    #             print ('LSTM XtAndInputCellAttention for {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName, epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmCellAtten, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmCellAtten,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break

    # print(">>>>>>>>>>>>> 1 layer XtAndInputCellAttention LSTM >>>>>>>>>>>>>>>>>>>>>>")

    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmCellAtten, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))


    # ########################################### Training Bi-LSTM ##########################################
    # print("Start Training" , modelName ,'Bi-LSTM')

    # netBiLstm = BiLSTM(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout).to(device)
    # netBiLstm.double()
    # optimizerBiLstm = torch.optim.Adam(netBiLstm.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_BiLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netBiLstm.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netBiLstm(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerBiLstm.zero_grad()
    #         loss.backward()
    #         optimizerBiLstm.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netBiLstm,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netBiLstm,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netBiLstm, saveModelBestName)
    #             print ('BiLSTM for {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netBiLstm, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netBiLstm,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer Bidirectional LSTM >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netBiLstm, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))





    # ########################################### XtAndInputCellAttention  Bidirectional LSTM ##########################################
    # print("Start Training" , modelName ,'Bidirectional XtAndInputCellAttention')

    # netBiLstmCellAtten= XtAndInputCellAttentionBiLstm(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout,args.attention_hops,args.d_a).to(device)
    # netBiLstmCellAtten.double()
    # optimizerBiLstmCellAtten = torch.optim.Adam(netBiLstmCellAtten.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_XtAndInputCellAttention_BiLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netBiLstmCellAtten.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netBiLstmCellAtten(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerBiLstmCellAtten.zero_grad()
    #         loss.backward()
    #         optimizerBiLstmCellAtten.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netBiLstmCellAtten,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netBiLstmCellAtten,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netBiLstmCellAtten, saveModelBestName)
    #             print ('BiLSTM XtAndInputCellAttention {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netBiLstmCellAtten, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netBiLstmCellAtten,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break

    # print(">>>>>>>>>>>>> 1 layer BiLSTM XtAndInputCellAttention >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(test_loaderRNN , netBiLstmCellAtten, args,Flag=True)
    # Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netBiLstmCellAtten,args)
    # print('BestEpochs {},BestTrainAcc {:.4f}, BestTestAcc{:.4f}'.format(BestEpochs , Train_Acc,Test_Acc))



    # ########################################### Training LSTM With Max Pooling ##########################################
    # print("Start Training" , modelName ,'LSTM With Max Pooling')



    # netUniLstmMaxPooling = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout , poolingType="max").to(device)
    # netUniLstmMaxPooling.double()
    # optimizerUniLstmMaxPooling = torch.optim.Adam(netUniLstmMaxPooling.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmMaxPooling.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmMaxPooling(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstmMaxPooling.zero_grad()
    #         loss.backward()
    #         optimizerUniLstmMaxPooling.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmMaxPooling,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmMaxPooling,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmMaxPooling, saveModelBestName)
    #             print ('LSTM With Max Pooling {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmMaxPooling, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmMaxPooling,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer  LSTM With Max Pooling >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmMaxPooling, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))

    # ########################################### Training InputCellAttention With Max Pooling ##########################################
    # print("Start Training" , modelName ,'LSTM InputCellAttention With Max Pooling')



    # netUniLstmMaxPoolingCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout, d_a=args.d_a,r=args.attention_hops,poolingType="max" , networkType="InputCellAttention").to(device)
    # netUniLstmMaxPoolingCellAtten.double()
    # optimizerUniLstmLMaxPoolingCellAtten = torch.optim.Adam(netUniLstmMaxPoolingCellAtten.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_InputCellAttention_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmMaxPoolingCellAtten.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmMaxPoolingCellAtten(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstmLMaxPoolingCellAtten.zero_grad()
    #         loss.backward()
    #         optimizerUniLstmLMaxPoolingCellAtten.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmMaxPoolingCellAtten,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmMaxPoolingCellAtten,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmMaxPoolingCellAtten, saveModelBestName)
    #             print ('LSTM InputCellAttention With Max Pooling {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmMaxPoolingCellAtten, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmMaxPoolingCellAtten,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer LSTM InputCellAttention With Max Pooling >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmMaxPoolingCellAtten, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))




    # ########################################### Training XtAndInputCellAttention With Max Pooling ##########################################
    # print("Start Training" , modelName ,'LSTM XtAndInputCellAttention With Max Pooling')



    # netUniLstmMaxPoolingCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout, d_a=args.d_a,r=args.attention_hops,poolingType="max" , networkType="XtAndInputCellAttention").to(device)
    # netUniLstmMaxPoolingCellAtten.double()
    # optimizerUniLstmLMaxPoolingCellAtten = torch.optim.Adam(netUniLstmMaxPoolingCellAtten.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_XtAndInputCellAttention_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmMaxPoolingCellAtten.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmMaxPoolingCellAtten(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstmLMaxPoolingCellAtten.zero_grad()
    #         loss.backward()
    #         optimizerUniLstmLMaxPoolingCellAtten.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmMaxPoolingCellAtten,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmMaxPoolingCellAtten,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmMaxPoolingCellAtten, saveModelBestName)
    #             print ('LSTM XtAndInputCellAttention With Max Pooling {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmMaxPoolingCellAtten, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmMaxPoolingCellAtten,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer LSTM XtAndInputCellAttention With Max Pooling >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmMaxPoolingCellAtten, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))









    # ########################################### Training LSTM With Mean Pooling ##########################################
    # print("Start Training" , modelName ,'LSTM With Mean Pooling')



    # netUniLstmMeanPooling = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout , poolingType="mean").to(device)
    # netUniLstmMeanPooling.double()
    # optimizerUniLstmMeanPooling = torch.optim.Adam(netUniLstmMeanPooling.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmMeanPooling.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmMeanPooling(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstmMeanPooling.zero_grad()
    #         loss.backward()
    #         optimizerUniLstmMeanPooling.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmMeanPooling,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmMeanPooling,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmMeanPooling, saveModelBestName)
    #             print ('LSTM With Mean Pooling {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmMeanPooling, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmMeanPooling,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer  LSTM With Mean Pooling >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmMeanPooling, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))

    # ########################################### Training InputCellAttention With Mean Pooling ##########################################
    # print("Start Training" , modelName ,'LSTM InputCellAttention With Mean Pooling')



    # netUniLstmMeanPoolingCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout, d_a=args.d_a,r=args.attention_hops,poolingType="mean" , networkType="InputCellAttention").to(device)
    # netUniLstmMeanPoolingCellAtten.double()
    # optimizerUniLstmLMeanPoolingCellAtten = torch.optim.Adam(netUniLstmMeanPoolingCellAtten.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_InputCellAttention_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmMeanPoolingCellAtten.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmMeanPoolingCellAtten(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstmLMeanPoolingCellAtten.zero_grad()
    #         loss.backward()
    #         optimizerUniLstmLMeanPoolingCellAtten.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmMeanPoolingCellAtten,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmMeanPoolingCellAtten,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmMeanPoolingCellAtten, saveModelBestName)
    #             print ('LSTM InputCellAttention With Mean Pooling {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmMeanPoolingCellAtten, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmMeanPoolingCellAtten,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer LSTM InputCellAttention With Mean Pooling >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmMeanPoolingCellAtten, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))










    # ########################################### Training XtAndInputCellAttention With Mean Pooling ##########################################
    # print("Start Training" , modelName ,'LSTM XtAndInputCellAttention With Mean Pooling')



    # netUniLstmMeanPoolingCellAtten = CustomRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout, d_a=args.d_a,r=args.attention_hops,poolingType="mean" , networkType="XtAndInputCellAttention").to(device)
    # netUniLstmMeanPoolingCellAtten.double()
    # optimizerUniLstmLMeanPoolingCellAtten = torch.optim.Adam(netUniLstmMeanPoolingCellAtten.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_XtAndInputCellAttention_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netUniLstmMeanPoolingCellAtten.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         outputs = netUniLstmMeanPoolingCellAtten(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizerUniLstmLMeanPoolingCellAtten.zero_grad()
    #         loss.backward()
    #         optimizerUniLstmLMeanPoolingCellAtten.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmMeanPoolingCellAtten,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmMeanPoolingCellAtten,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netUniLstmMeanPoolingCellAtten, saveModelBestName)
    #             print ('LSTM XtAndInputCellAttention With Mean Pooling {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netUniLstmMeanPoolingCellAtten, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netUniLstmMeanPoolingCellAtten,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break


    # print(">>>>>>>>>>>>> 1 layer LSTM XtAndInputCellAttention With Mean Pooling >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmMeanPoolingCellAtten, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))






    # ########################################### Training LSTM Self Attention ##########################################
    print("Start Training" , modelName ,'LSTM With SelfAttention')

    netUniLstmSelfAttention = StructuredSelfAttentionRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout , args.d_a, args.attention_hops ,args.sequence_length,"LSTM").to(device)
    netUniLstmSelfAttention.double()
    optimizerUniLstmSelfAttention = torch.optim.Adam(netUniLstmSelfAttention.parameters(), lr=args.learning_rate)


    saveModelName="../Models/"+modelName+"_UniLSTM_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)
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
            netUniLstmSelfAttention.train()
            samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
            samples = Variable(samples)
            labels = labels.to(device)
            labels = Variable(labels).long()

            # outputs,attention = netUniLstmL1SelfAttention(samples)
            outputs = netUniLstmSelfAttention(samples,seqLength)
            loss = criterion(outputs, labels)
            
            optimizerUniLstmSelfAttention.zero_grad()
            loss.backward()
            optimizerUniLstmSelfAttention.step()

            if (i+1) % 10 == 0:
                Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netUniLstmSelfAttention,args)
                Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netUniLstmSelfAttention,args)
                if(Test_Acc>BestAcc):
                    BestAcc=Test_Acc
                    BestEpochs = epoch+1
                    torch.save(netUniLstmSelfAttention, saveModelBestName)
                print ('LSTM With SelfAttention {}--> Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
                       .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


            if(epoch+1)%10==0:
                torch.save(netUniLstmSelfAttention, saveModelLastName)
            if(Train_Acc==100):
                torch.save(netUniLstmSelfAttention,saveModelLastName)
                Train_acc_flag=True
                break
        if(Train_acc_flag):
            break


    print(">>>>>>>>>>>>> 1 layer LSTM With SelfAttention >>>>>>>>>>>>>>>>>>>>>>")
    Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netUniLstmSelfAttention, args,Flag=True)
    print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))


    # # ########################################### Training LSTM Cell Atten Self Attention ##########################################
    # print("Start Training" , modelName ,'InputCellAttention With SelfAttention')

    # netInputCellAttentionSelfAttention = StructuredSelfAttentionRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout , args.d_a, args.attention_hops ,args.sequence_length,"InputCellAttention").to(device)
    # netInputCellAttentionSelfAttention.double()
    # optimizernetInputCellAttentionSelfAttention = torch.optim.Adam(netInputCellAttentionSelfAttention.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_InputCellAttention_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netInputCellAttentionSelfAttention.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         # outputs,attention = netUniLstmL1SelfAttention(samples)
    #         outputs = netInputCellAttentionSelfAttention(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizernetInputCellAttentionSelfAttention.zero_grad()
    #         loss.backward()
    #         optimizernetInputCellAttentionSelfAttention.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netInputCellAttentionSelfAttention,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netInputCellAttentionSelfAttention,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netInputCellAttentionSelfAttention, saveModelBestName)
    #             print ('InputCellAttention With SelfAttention {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netInputCellAttentionSelfAttention, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netInputCellAttentionSelfAttention,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break



    # print(">>>>>>>>>>>>> 1 layer InputCellAttention With SelfAttention >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netInputCellAttentionSelfAttention, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))

    # # ########################################### Training LSTM Self Attention ##########################################
    # print("Start Training" , modelName ,'XtAndInputCellAttention With SelfAttention')

    # netXtAndInputCellAttentionSelfAttention = StructuredSelfAttentionRNN(args.input_size, args.hidden_size1, args.num_layers, args.num_classes,args.rnndropout,args.LSTMdropout , args.d_a, args.attention_hops ,args.sequence_length,"XtAndInputCellAttention").to(device)
    # netXtAndInputCellAttentionSelfAttention.double()
    # optimizernetXtAndInputCellAttentionSelfAttention = torch.optim.Adam(netXtAndInputCellAttentionSelfAttention.parameters(), lr=args.learning_rate)


    # saveModelName="../Models/"+modelName+"_XtAndInputCellAttention_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a)
    # saveModelBestName =saveModelName +"_BEST.pkl"
    # saveModelLastName=saveModelName+"_LAST.pkl"
     
    

    # # Train the model
    # total_step = len(train_loaderRNN)
    # Train_acc_flag=False
    # Train_Acc=0
    # Test_Acc=0
    # BestAcc=0
    # BestEpochs = 0
    # for epoch in range(args.num_epochs):
    #     for i, (samples, labels,seqLength) in enumerate(train_loaderRNN):
    #         netXtAndInputCellAttentionSelfAttention.train()
    #         samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
    #         samples = Variable(samples)
    #         labels = labels.to(device)
    #         labels = Variable(labels).long()

    #         # outputs,attention = netUniLstmL1SelfAttention(samples)
    #         outputs = netXtAndInputCellAttentionSelfAttention(samples,seqLength)
    #         loss = criterion(outputs, labels)
            
    #         optimizernetXtAndInputCellAttentionSelfAttention.zero_grad()
    #         loss.backward()
    #         optimizernetXtAndInputCellAttentionSelfAttention.step()

    #         if (i+1) % 10 == 0:
    #             Test_Acc = checkAccuracyOnTestLstm(test_loaderRNN, netXtAndInputCellAttentionSelfAttention,args)
    #             Train_Acc = checkAccuracyOnTestLstm(train_loaderRNN, netXtAndInputCellAttentionSelfAttention,args)
    #             if(Test_Acc>BestAcc):
    #                 BestAcc=Test_Acc
    #                 BestEpochs = epoch+1
    #                 torch.save(netXtAndInputCellAttentionSelfAttention, saveModelBestName)
    #             print ('XtAndInputCellAttention With SelfAttention {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.5f}, Test Accuracy {:.5f},BestEpochs {},BestAcc {:.5f}' 
    #                    .format(args.DataName,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc))


    #         if(epoch+1)%10==0:
    #             torch.save(netXtAndInputCellAttentionSelfAttention, saveModelLastName)
    #         if(Train_Acc==100):
    #             torch.save(netXtAndInputCellAttentionSelfAttention,saveModelLastName)
    #             Train_acc_flag=True
    #             break
    #     if(Train_acc_flag):
    #         break



    # print(">>>>>>>>>>>>> 1 layer XtAndInputCellAttention With SelfAttention >>>>>>>>>>>>>>>>>>>>>>")
    # Train_Acc =checkAccuracyOnTestLstm(train_loaderRNN , netXtAndInputCellAttentionSelfAttention, args,Flag=True)
    # print('BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(BestEpochs , BestAcc , Train_Acc))
    os.system('say "your program has finished"')

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