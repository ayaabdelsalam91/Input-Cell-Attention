
import sys
sys.path.append("..")
from matplotlib import pyplot as plt
import numpy as np
import Helper
import math
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import re

Loc_Graph = '../Graphs/'
Results = '../Results/'


mainPaper=True
JaccardFlag=True
EucFlag=True
WeightedJaccardFlag=True
EsFlag=True
AucFlag=True
print("is mainPaper" , mainPaper)


if (mainPaper):
      modelsName_ = [
                "FF",
                "LSTM",
              
                "LSTM+cell At. Vector",
                "Bi-LSTM",
                "LSTM+self At.",
                  # "LSTM+cell At. Matrix"
                ]

      colors = [
                  "red",
                  "green",
                  "cyan",
                  "blue",
                  "purple",
                  "black" 
                  ]


      palecolors = [
                  "salmon",
                  "palegreen",
                  "lightcyan",
                  "lightblue",
                  "violet",
                  "lightgray"
                  ]

else:


      modelsName_ = [
                "LSTM",
                "LSTM+cell At.",
                "Bi-LSTM",
                "LSTM+Max pl",
                "LSTM+Max pl +cell At.",
                "LSTM+Mean pl",
                "LSTM+Mean pl +cell At.",
                "LSTM+self At.",
                "LSTM+self At.+cell At."]

      colors = [
                  "red",
                  "green",
                  "cyan",
                  "blue",
                  "purple" , 
                  "black" , 
                  "orange",
                  "maroon",
                  "deeppink"]


      palecolors = [
                  "salmon",
                  "palegreen",
                  "lightcyan",
                  "lightblue",
                  "violet",
                  "lightgray",
                  "wheat",
                  "lightcoral",
                  "pink"]
            


def getNumberOfMeasurmentBefore(name):
      num =0
      if(name=="Jac"):
            return num
      elif(name=="WJac"):
            if(JaccardFlag):
                  num+=1
      elif(name=="Euc"):
            if(JaccardFlag):
                  num+=1
            if(WeightedJaccardFlag):
                  num+=1
      elif(name=="Auc"):
            if(JaccardFlag):
                  num+=1
            if(WeightedJaccardFlag):
                  num+=1
            if(EucFlag):
                  num+=1
      elif(name=="Es"):
            if(JaccardFlag):
                  num+=1
            if(WeightedJaccardFlag):
                  num+=1
            if(EucFlag):
                  num+=1
            if(AucFlag):
                  num+=1
      return num
def getXYZ(modelName, columName):
      x=modelsName_.index(modelName)
     
      measurmentName = columName.split("_")[0]
      num = getNumberOfMeasurmentBefore(measurmentName)
      y=num*2
      if("_var" in columName):
            y+=1


      if("start = 0"in columName):
            z=0
      elif("start = 10"in columName):
            z=1      
      elif("start = 20"in columName):
            z=2 
      elif("start = 30"in columName):
            z=3 
      elif("start = 40"in columName):
            z=4 
      return x,y,z


measurmentsCount=0
saveName=""
MeasurmentAxies=[]
if(JaccardFlag):
      measurmentsCount+=1
      saveName+="_Jac"
      MeasurmentAxies.append("Jaccard")
if(WeightedJaccardFlag):
      measurmentsCount+=1
      saveName+="_WJac"
      MeasurmentAxies.append("Weighted Jaccard")
if(EucFlag):
      measurmentsCount+=1
      saveName+="_Euc"
      MeasurmentAxies.append("Euclidean Distance")
if(AucFlag):
      measurmentsCount+=1
      saveName+="_Auc"
      MeasurmentAxies.append("AUC")
if(EsFlag):
      measurmentsCount+=1
      saveName+="_Es"
      MeasurmentAxies.append("Enrichment Score")

if mainPaper:
      stat= Helper.load_CSV(Results+"MovingMiddleBoxStatsWith"+saveName+"Mat_MainPaper.csv" , returnDF=True)
else:
      stat= Helper.load_CSV(Results+"MovingMiddleBoxStatsWith"+saveName+"Mat_SupPaper.csv" , returnDF=True)
statCol = list(stat)
statData=stat.values
NumOfMeasurments=measurmentsCount*2
time=[i for i in range(0,50,10)]

measurements=np.zeros((len(modelsName_),NumOfMeasurments,len(time)))
for i in range(statData.shape[0]):
      for j in range(1,statData.shape[1]):
            x,y,z = getXYZ(statData[i,0], statCol[j])
            measurements[x,y,z]=statData[i,j]
            



if mainPaper:
  fig, axs = plt.subplots(1, measurmentsCount,sharex=True, figsize=(measurmentsCount*3+2,4))
else:
  fig, axs = plt.subplots(1, measurmentsCount,sharex=True, figsize=(measurmentsCount*4+3,8))

fig.suptitle('Effect of Moving Important Features on Different Measurements', fontsize=16)

ls=[0 for i in range(len(modelsName_))]



for i in range(len(modelsName_)):
      for j in range(0,NumOfMeasurments,2):
            k=int(j/2)
            # print(i ,colors[i] ,  modelsName_[i] , measurements[i,j,:].shape)
            if(mainPaper):
              ls[i] = axs[k].plot(time,measurements[i,j,:],'*-',color = colors[i] ,label=modelsName_[i])
            else:
              ls[i] = axs[k].plot(time,measurements[i,j,:],'-',color = colors[i] ,label=modelsName_[i])
            if(mainPaper):
                  axs[k].fill_between(time, measurements[i,j,:]-measurements[i,j+1,:], measurements[i,j,:]+measurements[i,j+1,:],color = palecolors[i])
            axs[k].set_title(MeasurmentAxies[k]  ,  fontsize=15)
            # axs[k].set_ylabel(MeasurmentAxies[k] ,  fontsize=13)
            axs[k].set_xlabel('Time' ,  fontsize=15)


lengendValues = []
for i in range(len(modelsName_)):
      lengendValues.append(ls[i][0])
if not mainPaper:
  x = 1- (2  / (measurmentsCount*4))-.1

  fig.legend(loc='center left', bbox_to_anchor=(x, 0.5), fontsize=13)
  fig.tight_layout(rect=[0, 0.03,x, 0.95])
else:
  fig.legend(loc='bottom left')
  fig.tight_layout(rect=[0, 0.03,1, 0.95])
#       fig.legend(loc='center left', bbox_to_anchor=(0.7, 0.5), fontsize=13)
#       fig.tight_layout(rect=[0, 0.03, 0.715, 0.95])

if mainPaper:
      plt.savefig(Loc_Graph+'effectOfChangingTimeOnMeasurementsMainPaper'+saveName+'.png')
else:
      plt.savefig(Loc_Graph+'effectOfChangingTimeOnMeasurementsSupPaper'+saveName+'.png')
plt.show()
