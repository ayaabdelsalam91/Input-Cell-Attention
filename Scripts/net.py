import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
import Helper
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from cell import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True

torch.manual_seed(999)




class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size1 , num_layers, num_classes , rnndropout , LSTMdropout,d_a=None,r=None,poolingType=None , networkType=None):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers = num_layers
        self.drop = nn.Dropout(rnndropout)  
        self.networkType=networkType
        self.poolingType=poolingType
        self.fc = nn.Linear(hidden_size1, num_classes) 
        if(poolingType!=None):
            if(poolingType=="max"):
                self.pooling = Helper.hiddenStateMaxPooling
            elif(poolingType=="mean"):
                self.pooling = Helper.hiddenStateMeanPooling
            print(self.poolingType)

        if(networkType!=None):
            if(networkType =="LSTM"): 
                self.rnn = nn.LSTM(input_size,hidden_size1,self.num_layers,batch_first=True)
            elif(networkType =="GRU"): 
                self.rnn = nn.GRU(input_size, hidden_size1,self.num_layers,  batch_first=True)
            elif(networkType =="RNN"): 
                self.rnn = nn.RNN(input_size, hidden_size1,self.num_layers,  batch_first=True)
            elif(networkType=="InputCellAttention"):
                self.rnn =LSTMWithInputCellAttention(input_size, hidden_size1,r,d_a)
        else:
            self.rnn = nn.LSTM(input_size,hidden_size1,self.num_layers,batch_first=True)
            self.networkType="LSTM"


    def forward(self, x,X_lengths):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device)
        h0 = h0.double()
        c0 = c0.double()
        x = self.drop(x)

        if(self.networkType!="RNN"):
            output, _ = self.rnn(x, (h0, c0))
        else:
            output, _ = self.rnn(x, h0)


   

        output = self.drop(output)
        if(self.poolingType!=None):
            output = self.pooling(output)
        else:
            idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
            time_dimension = 1 if batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if output.is_cuda:
                idx = idx.cuda(output.data.get_device())
            output = output.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out
