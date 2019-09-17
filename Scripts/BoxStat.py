import torch
from net import *
import Helper
import argparse
import random
import sys
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from net import *
import random
from customLSTM import *
from sklearn import metrics 
from accuracyMethods import *
from scipy.linalg import norm
from scipy.stats import pearsonr

mainPaper=True
JaccardFlag=False
EucFlag=True
WeightedJaccardFlag=True
EsFlag=False
AucFlag=False
ACCFlag=False
print("is mainPaper" , mainPaper)

Results='../Results/'


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



def main(args):

    
    boxes = ["TopBox"  ,     "MiddleBox"  ,   "BottomBox"   ,  "MixedBoxes"  ,    "ThreeUpperBoxes"  ,    "ThreeMiddleBoxes"]

    printboxes = ["Ealier Box"  ,     "Middle Box"  ,   "Later Box"  ,   "Mixed Boxes"    ,  "3 Ealier Boxes"  ,    "3 Middle Boxes"]
    
    startImportance = [  0 , 20,70, ''  , [0,0,0], [25,25,25]]


    
    if (mainPaper):
        modelsName_ = [
                  "LSTM",
                  "Bi-LSTM",
                  "LSTM+self At.",
                  "LSTM+cell At. Vector",
                 "LSTM+cell At. Matrix",

                  ]

    else:

        modelsName_ = [
                    "LSTM",
                    "Bi-LSTM",
                    "LSTM+Max pl",
                    "LSTM+Max pl +cell At.",
                    "LSTM+Mean pl",
                    "LSTM+Mean pl +cell At.",
                    "LSTM+self At.",
                    "LSTM+self At.+cell At.",
                    "LSTM+cell At."]



    measurmentsCount=0
    saveName=""
    if(JaccardFlag):
        measurmentsCount+=1
        saveName+="_Jac"
    if(WeightedJaccardFlag):
        measurmentsCount+=1
        saveName+="_WJac"
    if(EucFlag):
        measurmentsCount+=1
        saveName+="_Euc"
    if(AucFlag):
        measurmentsCount+=1
        saveName+="_Auc"
    if(EsFlag):
        measurmentsCount+=1
        saveName+="_Es"
    if (ACCFlag):
        measurmentsCount+=1
        saveName+="_ACC"

    stats=np.zeros((len(modelsName_),1+measurmentsCount*len(boxes)),dtype=object)
    for i in range(len(modelsName_)):
        stats[i,0] = modelsName_[i]
    cols =["Model"]
    jump=0


    for b in range(len(boxes)):
        args.DataName=boxes[b]
        args.importance=startImportance[b]

        modelName = "Simulated"
        modelName+=args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)

        TestingLabel = Helper.load_CSV(args.data_dir+"SimulatedTestingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
        Testing = Helper.load_CSV(args.data_dir+"SimulatedTestingData"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")

        TestingSeqLength= TestingLabel[:,1].astype(int)
        TestingType=TestingLabel[:,-1]
        Types = np.unique(TestingType)
        TestingLabel= TestingLabel[:,0]
        TestingLabel=TestingLabel.reshape(TestingLabel.shape[0],)
        TestingLabel = Helper.reOrderLabels(TestingLabel.tolist())
        TestingLabel=np.array(TestingLabel)


        if(ACCFlag):

            scaler = StandardScaler()
            scaler.fit(Testing)
            Testing = scaler.transform(Testing)

            TestingRNN = Testing.reshape(Testing.shape[0] , args.sequence_length,args.input_size)
          
            test_dataRNN = data_utils.TensorDataset(torch.from_numpy(TestingRNN),torch.from_numpy( TestingLabel) , torch.from_numpy(TestingSeqLength))
            test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)



        if mainPaper:

            models=[
                        
                        "../Models/"+modelName+"_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                        "../Models/"+modelName+"_BiLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                        "../Models/"+modelName+"_UniLSTM_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                        "../Models/"+modelName+"_InputCellAttention_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),
                        "../Models/"+modelName+"_InputCellAttentionMat_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),

                        ]
        else:
                    models=[
                    
                    "../Models/"+modelName+"_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                    "../Models/"+modelName+"_BiLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                    
                    "../Models/"+modelName+"_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                    "../Models/"+modelName+"_InputCellAttention_UniLSTM_MAXPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),

                    "../Models/"+modelName+"_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                    "../Models/"+modelName+"_InputCellAttention_UniLSTM_MEANPOOLING_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                    
                    "../Models/"+modelName+"_UniLSTM_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes),
                    "../Models/"+modelName+"_InputCellAttention_SelfAttention_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),
                    
                    "../Models/"+modelName+"_InputCellAttention_UniLSTM_L"+ str(args.num_layers) +"_"+str(args.hidden_size1)+"_Scaled_"+ str(args.learning_rate)+"_DrpRNN"+str(args.rnndropout)+"_DrpLSTM"+str(args.LSTMdropout)+'_'+str(args.num_classes)+"_r"+str(args.attention_hops)+"_da"+str(args.d_a),

                    ]

        if(args.DataName=="MixedBoxes"):
                importantInd=[]
                impFeatures=[]
                referenceSamples=[]
                for i in range(1,8):
                    referenceSample , indx ,numberOfImpFeatures= getIndexOfImpValues(i)
                    importantInd.append(indx)
                    impFeatures.append(numberOfImpFeatures)
                    referenceSamples.append(referenceSample)
        elif(args.DataName=="MovingMiddleBox"):
            referenceSample , referenceIndx ,impFeaturesCount =getIndexOfImpValues(args.DataName,args.importance)

        else:
            referenceSample , referenceIndx ,impFeaturesCount =getIndexOfImpValues(args.DataName)


        for m , model in enumerate(models):


            statCount=1

            ModelType = getModelType(model,args)


            saliencies = Helper.load_CSV(Results+args.DataName+ModelType+str(args.importance)+"saliencies.csv")

            if(ACCFlag):
                model=model+"_BEST.pkl"
                pretrained_model = torch.load(model,map_location='cpu') 
                Test_Acc  =   Helper.checkAccuracyOnTestLstm(test_loaderRNN , pretrained_model, args,Flag=True)
                print(ModelType , printboxes[b] ,  Test_Acc)



            
            if(JaccardFlag):
                JaccardScores=np.zeros((TestingLabel.shape[0]))
            if(EucFlag):
                euclideanScores=np.zeros((TestingLabel.shape[0]))
            if(WeightedJaccardFlag):
                WeightedJaccardScores=np.zeros((TestingLabel.shape[0]))
            if(EsFlag):
                EnrichmentScores=np.zeros((TestingLabel.shape[0]))
            if(AucFlag):
                AUCScores=np.zeros((TestingLabel.shape[0]))



            stats[m,0]=modelsName_[m]
            for i in range (TestingLabel.shape[0]):

                if(args.DataName=="MixedBoxes"):
                    referenceIndx=importantInd[int(TestingType[i]-1)]
                    impFeaturesCount=impFeatures[int(TestingType[i]-1)]
                    referenceSample = referenceSamples[int(TestingType[i]-1)]


                saliencyIndex =  getIndexOfMaxValues(saliencies[i],impFeaturesCount)
                saliency =saliencies[i]
                rescaledSaliency= rescale_(saliency)
                rescaledSample=rescale_(np.abs(Testing[i]))
                referenceSample_=referenceSample.reshape(args.sequence_length*args.input_size)

                if(JaccardFlag):
                    JaccardScores[i]=getJaccardSimilarityScore(referenceIndx,saliencyIndex)
                if(WeightedJaccardFlag):
                    WeightedJaccardScores[i]=getWeightedJaccardSimilarityScore(referenceSample_,rescaledSaliency)
                if(EsFlag):
                    EnrichmentScores[i]=getEnrichmentScore(referenceSample,rescaledSaliency,referenceIndx,saliencyIndex,impFeaturesCount)
                if(AucFlag):
                    AUCScores[i] = metrics.roc_auc_score(referenceSample_ , rescaledSaliency )
                if(EucFlag):
                    rescaledSaliency = rescaledSaliency.reshape(args.sequence_length,args.input_size)
                    euclideanScores[i]= np.linalg.norm(referenceSample-rescaledSaliency)/np.linalg.norm(referenceSample)
      



            if(JaccardFlag):
                stats[m,statCount+jump]=np.mean(JaccardScores)
                statCount+=1
    

            if(WeightedJaccardFlag):
                stats[m,statCount+jump]=np.mean(WeightedJaccardScores)
                statCount+=1

            if(EucFlag):
                stats[m,statCount+jump]=np.mean(euclideanScores)
                statCount+=1


            if(AucFlag):
                stats[m,statCount+jump]=np.mean(AUCScores)
                statCount+=1


            if(EsFlag):
                stats[m,statCount+jump]=np.mean(EnrichmentScores)
                statCount+=1

            if(ACCFlag):
                stats[m,statCount+jump]=Test_Acc
                statCount+=1



        jump+=measurmentsCount
        print(printboxes[b])
        if(JaccardFlag):
            cols.append("Jac "+printboxes[b])
        if(WeightedJaccardFlag):
            cols.append("WJac "+printboxes[b])
            
        if(EucFlag):
            cols.append("Euc "+printboxes[b])

        if(AucFlag):
            cols.append("Auc "+printboxes[b])

        if(EsFlag):
            cols.append("Es "+printboxes[b])

        if(ACCFlag):
            cols.append("ACC "+printboxes[b])


    if(mainPaper):
        Helper.save_intoCSV(stats,Results+"BoxStatsWith"+saveName+"_MainPaper.csv",col=cols)
    else:
        Helper.save_intoCSV(stats,Results+"BoxStatsWith"+saveName+"_SupPaper.csv",col=cols)




  
def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--importance', type=str, default=70)
    parser.add_argument('--DataName', type=str ,default="BottomBox")
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