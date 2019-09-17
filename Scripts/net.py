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


class StructuredSelfAttentionRNN(nn.Module):
    def __init__(self, input_size, hid_dim , num_layers, num_classes , dropout , RNNdropout,d_a,r ,max_len,networkType,type=0):
        super(StructuredSelfAttentionRNN, self).__init__()

        """
        Initializes parameters suggested in paper
 
        Args:
            input_size  : {int} input dimension
            hid_dim     : {int} hidden dimension
            num_layers  : {int} number of layers
            num_classes : {int} number of classes
            dropout     : {Float} Dropout for input
            RNNdropout : {Float} Dropout for recurrent model
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of timesteps
            networkType : {String} recurrent network type "LSTM" ,"GRU" or "RNN"
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            
 
        Returns:
            self
 
        Raises:
            Exception
        """
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        if(networkType =="LSTM"): 
            self.rnn = nn.LSTM(input_size,hid_dim,self.num_layers,batch_first=True)
        elif(networkType =="GRU"): 
            self.rnn = nn.GRU(input_size, hid_dim,self.num_layers,  batch_first=True)
        elif(networkType =="RNN"): 
            self.rnn = nn.RNN(input_size, hid_dim,self.num_layers,  batch_first=True)
        elif(networkType=="InputCellAttention"):
            self.rnn =LSTMWithCellAttentionVector(input_size, hid_dim,r,  d_a)
        elif(networkType=="XtAndInputCellAttention"):
            self.rnn=LSTMWithXtAndCellAttentionVector(input_size, hid_dim,r,  d_a)
        elif(networkType=="hiddenCellAttention"):
            self.rnn =LSTMWithHiddenAttention(input_size, hid_dim,r,  d_a)



        self.linear_first = torch.nn.Linear(hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.n_classes = num_classes
        self.linear_final = torch.nn.Linear(hid_dim,self.n_classes)
        self.max_len = max_len
        self.hid_dim = hid_dim
        self.r = r
        self.type = type
        self.networkType=networkType
        

    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def forward(self, x,X_lengths):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hid_dim).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hid_dim).to(device)

        h0 = h0.double()
        c0 = c0.double()
        x = self.drop(x)
        if(self.networkType=="LSTM" or self.networkType=="InputCellAttention" or self.networkType=="hiddenCellAttention" or self.networkType=="XtAndInputCellAttention"):
            outputs, _ = self.rnn(x, (h0, c0))
        else:
            outputs, _ = self.rnn(x, h0)

        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)  
        x = self.softmax(x,1)       
        attention = x.transpose(1,2)   
        sentence_embeddings = attention@outputs       
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
       
        if not bool(self.type):
            output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
            # return output,attention
            return output
        else:
            # return F.log_softmax(self.linear_final(avg_sentence_embeddings)),attention
            return F.log_softmax(self.linear_final(avg_sentence_embeddings))


class FFL1(nn.Module):
    def __init__(self, input_size, hidden_size1 ,dropout ,num_classes):
        super(FFL1, self).__init__()                   
        self.fc1 = nn.Linear(input_size, hidden_size1)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size1,num_classes)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):                            
        x = self.drop(x)
        out = self.fc1(x)
        out = self.relu(out)
        last_output=self.drop(out)
        out = self.fc2(last_output)
        out =F.softmax(out, dim=1)
        return out


# class UniLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size1 , num_layers, num_classes , rnndropout , LSTMdropout):
#         super().__init__()
#         self.hidden_size1 = hidden_size1
#         self.num_layers = num_layers
#         self.drop = nn.Dropout(rnndropout)  
#         self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers ,   batch_first=True)

#         self.fc = nn.Linear(hidden_size1, num_classes) 
#     def forward(self, x,X_lengths):
#         # Set initial states
#         h0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device) 
#         c0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device)
#         h0 = h0.double()
#         c0 = c0.double()
#         x = self.drop(x)
#         output, _ = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
#         # print(list(self.lstm1.parameters()))
#         output = self.drop(output)
#         idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
#             len(X_lengths), output.size(2))
#         time_dimension = 1 if batch_first else 0
#         idx = idx.unsqueeze(time_dimension)
#         if output.is_cuda:
#             idx = idx.cuda(output.data.get_device())
#         last_output = output.gather(
#             time_dimension, Variable(idx)).squeeze(time_dimension)
#         out = self.fc(last_output)
#         out =F.softmax(out, dim=1)
#         return out




class UniLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1 , num_layers, num_classes , rnndropout , LSTMdropout):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers = num_layers
        self.drop = nn.Dropout(rnndropout)  
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers ,   batch_first=True).to(device) 

        self.fc = nn.Linear(hidden_size1, num_classes) 
    def forward(self, x,X_lengths):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device)
        h0 = h0.double()
        c0 = c0.double()
        x = self.drop(x)
        output, _ = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # print(list(self.lstm1.parameters()))
        output = self.drop(output)
        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
        time_dimension = 1 if batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        out = self.fc(last_output)
        out =F.softmax(out, dim=1)
        return out



# Bidirectional recurrent neural network (many-to-one)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes , rnndropout , LSTMdropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(rnndropout)
        if(num_layers==1):
             self.lstm = nn.LSTM(input_size, hidden_size, num_layers , batch_first=True, bidirectional=True)
        else:      
             self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout =  LSTMdropout,  batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x,X_lengths):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        h0 = h0.double()
        c0 = c0.double()
        x = self.drop(x)
        output, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        output = self.drop(output)
        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
        time_dimension = 1 if batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
      
        out = self.fc(last_output)
        return out





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
            elif(networkType=="InputCellAttentionVector"):
                self.rnn =LSTMWithCellAttentionVector(input_size, hidden_size1,r,  d_a)
            elif(networkType=="InputCellAttentionMatrix"):
                self.rnn =LSTMWithCellAttentionMatrix(input_size, hidden_size1,r,  d_a)

            elif(networkType=="XtAndInputCellAttention"):
                self.rnn =LSTMWithXtAndCellAttentionVector(input_size, hidden_size1,r,  d_a)
            elif(networkType=="hiddenCellAttention"):
                self.rnn =LSTMWithHiddenAttention(input_size, hidden_size1,r,  d_a)
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





class UniGRU(nn.Module):
    def __init__(self, input_size, hidden_size1 , num_layers, num_classes , rnndropout , LSTMdropout):
        super(UniGRUL1, self).__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers = num_layers
        self.drop = nn.Dropout(rnndropout)  

        self.gru1 = nn.GRU(input_size, hidden_size1, num_layers ,   batch_first=True)

        self.fc = nn.Linear(hidden_size1, num_classes) 
    def forward(self, x,X_lengths):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size1).to(device) 
        h0 = h0.double()

        x = self.drop(x)
        output, _ = self.gru1(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        output = self.drop(output)
        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
        time_dimension = 1 if batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        out = self.fc(last_output)
        out =F.softmax(out, dim=1)
        return out




class UniLstmCellL1(nn.Module):
    def __init__(self, input_size, hidden_size1 , num_layers, num_classes , rnndropout , LSTMdropout):
        super(UniLstmCellL1, self).__init__()
        self.hidden_size1 = hidden_size1

        self.num_layers = num_layers
        self.drop = nn.Dropout(rnndropout)  

        self.lstm1 = nn.LSTMCell(input_size, hidden_size1)

        self.fc = nn.Linear(hidden_size1, num_classes) 
    def forward(self, x,X_lengths):
        # Set initial states
        ht_cell = torch.zeros(x.size(1),  self.hidden_size1).to(device) 
        ct_cell = torch.zeros(x.size(1), self.hidden_size1).to(device)
        ht_cell = ht_cell.double()
        ct_cell = ct_cell.double()
        x = self.drop(x)
        out_cell=[]
        states_cell=[]
        for i in range(x.size(0)):
            # print(x[i].shape , ht_cell.shape,ct_cell.shape)

            ht_cell,ct_cell = self.lstm1(x[i],(ht_cell,ct_cell))
            out_cell.append(ht_cell)
            states_cell.append(ct_cell)

        output, _  =  torch.stack(out_cell), torch.stack(states_cell)

        output = self.drop(output)
        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
        time_dimension = 1

        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        output=output.view(x.size(0),x.size(1),output.size(2))
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)

        out = self.fc(last_output)
        out =F.softmax(out, dim=1)
        return out


class UniRNN(nn.Module):
    def __init__(self, input_size, hidden_size , num_layers, num_classes , rnndropout):
        super(UniRNNL1, self).__init__()


        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout(rnndropout)  
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.hidden_size = hidden_size
    
    def forward(self, x, X_lengths):

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        h0 = h0.double()
        x = self.drop(x)
        output, _ = self.rnn(x, h0)
        output = self.drop(output)

        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)

        out = self.fc(last_output)
        out =F.softmax(out, dim=1)
        return out





class UniRNNCellL1(nn.Module):
    def __init__(self, input_size, hidden_size , num_layers, num_classes , rnndropout ):
        super(UniRNNCellL1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(rnndropout)  
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes) 

    def forward(self, x,X_lengths):


        # Set initial states
        ht_cell = torch.zeros(x.size(0),  self.hidden_size).to(device) 
        ht_cell = ht_cell.double()
        x = self.drop(x)
        out_cell=[]

        for i in range(x.size(1)):
            ht_cell = self.rnn(x[:,i,:],ht_cell)
            out_cell.append(ht_cell)

        output  =  torch.stack(out_cell)
        output = self.drop(output)
        # print(output.shape)
        # print(ht_cell.shape)

        output = output.permute(1,0,2)
        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), output.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)

        out = self.fc(last_output)
        out =F.softmax(out, dim=1)
        return out




class SimpleRNNAveragedH(nn.Module):
    def __init__(self, input_size, hidden_size , num_classes , seq_length ):
        super(SimpleRNNAveragedH, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(seq_length, num_classes) 
    def forward(self, x,X_lengths):
        # Set initial states
        ht_cell = torch.zeros(x.size(1),  self.hidden_size).to(device)
        ht_cell = ht_cell.double()
        out_cell=[]
        for i in range(x.size(0)):
            ht_cell = self.rnn(x[i],ht_cell)
            out_cell.append(ht_cell)
        output  =  torch.stack(out_cell)
        output=output.view(x.size(1),x.size(0)*output.size(2))

        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out



class SimpleRNNStackedH(nn.Module):
    def __init__(self, input_size, hidden_size , num_classes , seq_length ):
        super(SimpleRNNStackedH, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(seq_length*hidden_size, num_classes) 
    def forward(self, x,X_lengths):
        # Set initial states
        ht_cell = torch.zeros(x.size(1),  self.hidden_size).to(device)
        ht_cell = ht_cell.double()
        out_cell=[]
        for i in range(x.size(0)):
            ht_cell = self.rnn(x[i],ht_cell)
            out_cell.append(ht_cell)
        output  =  torch.stack(out_cell)
        output=output.view(x.size(1),x.size(0)*output.size(2))

        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out

class RNN_StackedH(nn.Module):
    def __init__(self, input_size, hidden_size , num_layers, num_classes , rnndropout ,  seq_length):
        super(RNN_StackedH, self).__init__()


        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout(rnndropout)  
        self.fc = nn.Linear(seq_length*hidden_size, num_classes) 
        self.hidden_size = hidden_size
    
    def forward(self, x, X_lengths):

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        h0 = h0.double()
        x = self.drop(x)
        output, _ = self.rnn(x, h0)
        output = self.drop(output)
        output = output.contiguous().view(output.size(0),output.size(1)*output.size(2))

        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out


class RNN_AveragedH(nn.Module):
    def __init__(self, input_size, hidden_size , num_layers, num_classes , rnndropout ):
        super(RNN_AveragedH, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout(rnndropout)  
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.hidden_size = hidden_size
    
    def forward(self, x, X_lengths):

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        h0 = h0.double()
        x = self.drop(x)
        output, _ = self.rnn(x, h0)
        output = self.drop(output)
        # print(output.shape)
        output = torch.sum(output, dim=1)
        # print(output.shape)
        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out

