import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
import Helper
import numpy as np


import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V
import time


import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

from typing import *
from pathlib import Path

import torch
import torch.nn as nn 
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True



from enum import IntEnum
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
 

class OptimizedLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size


        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias

            # print(x_t.shape ,  gates.shape)

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.squeeze()

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTMWithMaxXBar(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_iBarh = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

         
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size

        xBar=torch.zeros(x[:, 0, :].size()).double()
        TS=torch.zeros(x[:, 0, :].size()).double()
        one_=torch.ones(x[:, 0, :].size()).double()

        for t in range(seq_sz):


            x_t = x[:, t, :]
            if(t>1):
                xBar = Helper.hiddenStateMaxPooling(x[:, :t-1, :])

            # print(xBar)
            # print(x_t)
            # print("")
            print(xBar.shape)
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

            # print(gates.shape)
            # print((x_t @ self.weight_ih).shape , (xBar @ self.weight_iBarh).shape, (h_t @ self.weight_hh).shape)

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            # self.update_xBar(x_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

            # xBar=(xBar*TS)
            # TS=TS+one_
            # xBar=(xBar+x_t)/TS

     

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)

        hidden_seq = hidden_seq.squeeze(1)
        # print(len(hidden_seq.size()) , hidden_seq.size())

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTMWithMeanXBar(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_iBarh = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

         
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size

        xBar=torch.zeros(x[:, 0, :].size()).double()
        TS=torch.zeros(x[:, 0, :].size()).double()
        one_=torch.ones(x[:, 0, :].size()).double()
        # print(self.weight_iBarh)
        # print(self.weight_ih)
        for t in range(seq_sz):


            x_t = x[:, t, :]
            # print(xBar)
            # print(x_t)
            # print("")

            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            # self.update_xBar(x_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

            xBar=(xBar*TS)
            TS=TS+one_
            xBar=(xBar+x_t)/TS

     

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)

        hidden_seq = hidden_seq.squeeze(1)
        # print(len(hidden_seq.size()) , hidden_seq.size())

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTMWithHiddenAttention(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hBarh = Parameter(torch.Tensor(r* hidden_sz,  hidden_sz* 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r=r
        self.linear_first = torch.nn.Linear(hidden_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)

        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def getMatrixM(self, pastTimeSteps):
        # print("pastTimeSteps",pastTimeSteps.shape)

        x= self.linear_first(pastTimeSteps)

        # for param in self.linear_first.parameters():
        #     print("WS1",param.name , param.data.shape)


        x = torch.tanh(x)


        x = self.linear_second(x) 
        # for param in self.linear_second.parameters():
        #     print("WS2",param.name , param.data.shape)
        # print("after WS2" , x.shape) 
        x = self.softmax(x,1)

        attention = x.transpose(1,2) 
        # print("attention" , attention.shape)   
        matrixM = attention@pastTimeSteps 
        # print("matrixM" , matrixM.shape) 
        return matrixM

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


    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        batchSize=x[:, 0, :].size()[0]

        M=torch.zeros(batchSize , self.r , self.hidden_size).double().to(device) 
        H=torch.zeros(batchSize,1,self.hidden_size).double().to(device) 
        for t in range(seq_sz):
            


            x_t = x[:, t, :]
            # print(t, x_t.shape)

            # if(t==0):
            #     H=x[:, 0, :].view(batchSize,1,self.input_sz)
            #     print(H.shape)
            #     H=np.zeros((batchSize,1,self.hidden_size))
            #     print(H.shape)
            #     M = self.getMatrixM(H)
            if(t>0):
                H=hidden_seq_
            M = self.getMatrixM(H)

            # print(M.shape , self.r  , self.hidden_size)
                

            hBar=M.view(batchSize,self.r*self.hidden_size)
            # print(t,"x_t.shape",x_t.shape,"Rest" ,(x_t @ self.weight_ih).shape , (hBar @ self.weight_hBarh).shape)

        #     # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + hBar @ self.weight_hBarh+ self.bias
            #gates = xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias
            gates=gates.unsqueeze(0)
            # print(t,gates.shape)

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            # self.update_xBar(x_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
            hidden_seq_= torch.cat(hidden_seq, dim=Dim.batch)
            hidden_seq_=  hidden_seq_.squeeze(1)
            hidden_seq_ = hidden_seq_.transpose(Dim.batch, Dim.seq).contiguous()
            # print(t,"hidden_seq_.shape" , hidden_seq_.shape)



  

     

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)

        hidden_seq = hidden_seq.squeeze(1)
        # print(len(hidden_seq.size()) , hidden_seq.size())
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTMWithCellAttentionVector(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        print("LSTMWithCellAttentionVector")
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_iBarh = Parameter(torch.Tensor(r* input_sz,  hidden_sz* 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r=r
        self.linear_first = torch.nn.Linear(input_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.softmax_=nn.Softmax()
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def getMatrixM(self, pastTimeSteps):
        x= self.linear_first(pastTimeSteps)
        x = F.tanh(x)
        x = self.linear_second(x) 
        attention = self.softmax_(x)
        attention=attention.unsqueeze(2)
        pastTimeSteps=pastTimeSteps.unsqueeze(1)
        matrixM = attention@pastTimeSteps 
        return matrixM

    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        batchSize=x[:, 0, :].size()[0]

        M=torch.zeros(batchSize , self.r , self.input_sz).double()

        for t in range(seq_sz):

            x_t = x[:, t, :]
            if(t==0):
                H=x[:, 0, :]
                M = self.getMatrixM(H)
               
            elif(t>0):
                H=x[:, t, :]
                newM=self.getMatrixM(H)
                M = (M+newM)

            xBar=M.view(batchSize,self.r*self.input_sz)
            xBar = F.normalize(xBar, p=2, dim=1)
            gates = xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias
            

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

  

     

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)

        hidden_seq = hidden_seq.squeeze(1)

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)




class LSTMWithXtAndCellAttentionVector(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        print("LSTMWithXtAndCellAttentionVector")
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        # torch.manual_seed(666)
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        # torch.manual_seed(666)
        self.weight_iBarh = Parameter(torch.Tensor(r* input_sz,  hidden_sz* 4))
        # torch.manual_seed(666)
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        # torch.manual_seed(666)
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))

        self.r=r
        # torch.manual_seed(666)
        self.linear_first = torch.nn.Linear(input_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        # torch.manual_seed(666)

        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)

        self.softmax_=nn.Softmax()
        # # self.beta = torch.autograd.Variable(torch.rand(1).to(device), requires_grad=True).double()
        # self.beta=0.3
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def getMatrixM(self, pastTimeSteps):        
        x= self.linear_first(pastTimeSteps)
        x = F.tanh(x)
        x = self.linear_second(x) 
        attention = self.softmax_(x)
        attention=attention.unsqueeze(2)
        pastTimeSteps=pastTimeSteps.unsqueeze(1)
        matrixM = attention@pastTimeSteps 
        # print(matrixM.shape, "matrixM")
        return matrixM

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
        # print("LSTMWithAttentionXBar" , soft_max_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)


    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        batchSize=x[:, 0, :].size()[0]

        M=torch.zeros(batchSize , self.r , self.input_sz).double()

        for t in range(seq_sz):
            # print("OptimizedLSTMWithAttentionXBar t=" ,t )

            x_t = x[:, t, :]
            if(t==0):
                H=x[:, 0, :]
                M = self.getMatrixM(H)
               
            elif(t>0):
                H=x[:, t, :]
                newM=self.getMatrixM(H)
                M = (M+newM)
                # if(self.beta<1):
                #     M = self.beta*M+(1-self.beta)*newM


                # print(newM)

                
            # print(M.shape)
            xBar=M.view(batchSize,self.r*self.input_sz)
            xBar = F.normalize(xBar, p=2, dim=1)
            # gates = x_t @ self.weight_ih + xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias
            gates = x_t @ self.weight_ih + xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias
            

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            # print(gates)
            # print("i_t" , i_t)
            # print("f_t" ,f_t)
            # print("g_t" ,g_t)
            # print("o_t" ,o_t)  
            # print("h_t" ,h_t)  
            # print("")  

            # print(t,"h_t.shape" , h_t.unsqueeze(Dim.batch).shape)
            # self.update_xBar(x_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

  

     

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)

        hidden_seq = hidden_seq.squeeze(1)
        # print(len(hidden_seq.size()) , hidden_seq.size())

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTMWithCellAttentionMatrix(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_iBarh = Parameter(torch.Tensor(r* input_sz,  hidden_sz* 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r=r
        self.softmax_=nn.Softmax()
        self.linear_first = torch.nn.Linear(input_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.init_weights()
        print("LSTMWithCellAttentionMatrix")
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def getMatrixM(self, pastTimeSteps):

        x= self.linear_first(pastTimeSteps)

        x = F.tanh(x)
        x = self.linear_second(x) 
        x = self.softmax(x,1)
        attention = x.transpose(1,2) 
        matrixM = attention@pastTimeSteps 
        return matrixM

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


    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        batchSize=x[:, 0, :].size()[0]

        M=torch.zeros(batchSize , self.r , self.input_sz).double()

        for t in range(seq_sz):
            x_t = x[:, t, :]
            if(t==0):
                H=x[:, 0, :].view(batchSize,1,self.input_sz)

                M = self.getMatrixM(H)
            elif(t>0):
                H=x[:, :t+1, :]

                M = self.getMatrixM(H)




            xBar=M.view(batchSize,self.r*self.input_sz)
            xBar = F.normalize(xBar, p=2, dim=1)
            gates = xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)

        hidden_seq = hidden_seq.squeeze(1)

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class OptimizedLSTMWithAttentionXBar_(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,r:int,d_a:int , bias=True, dropout=0.0, dropout_method='pytorch',learnable=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(r* input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method
        self.r=r
        self.linear_first = torch.nn.Linear(input_size,d_a)
        self.linear_first.bias.data.fill_(0)

        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)

        self.softmax_=nn.Softmax()

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    # def getMatrixM(self, pastTimeSteps):
    #     x= self.linear_first(pastTimeSteps)
    #     x = F.tanh(x)
    #     x = self.linear_second(x) 
    #     attention = self.softmax_(x)
    #     attention=attention.unsqueeze(2)
    #     pastTimeSteps=pastTimeSteps.unsqueeze(1)
    #     matrixM = attention@pastTimeSteps 
    #     return matrixM

    def getMatrixM(self, pastTimeSteps):

        x= self.linear_first(pastTimeSteps)

        x = F.tanh(x)
        x = self.linear_second(x) 
        x = self.softmax(x,1)
        attention = x.transpose(1,2) 
        matrixM = attention@pastTimeSteps 
        return matrixM

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

    def forward(self, x, hidden):
        bs, seq_sz, _ = x.size()
        print( bs, seq_sz, _)
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        # x = x.view(x.size(1), -1)
        M=self.getMatrixM(x)
        xBar=M.view(bs,self.r*self.input_size)

        # Linear mappings
        preact = self.i2h(xBar) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]


        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)
        h_t = th.mul(o_t, c_t.tanh())
        h_t = h_t.view( h_t.size(0),1, -1)
        c_t = c_t.view( c_t.size(0),1, -1)

        return h_t, (h_t, c_t)







