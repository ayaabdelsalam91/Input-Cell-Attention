

import Helper
import argparse
import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors

Loc_Graph = '../Graphs/'
Results='../Results/'


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plotHeatMapExampleWise(input,title, saveLocation,greyScale=False,flip=False,x_axis=None,y_axis=None,show=True):
            
    if(flip):
        input=np.transpose(input)
    fig, ax = plt.subplots()

    if(greyScale):
        cmap='gray'
    else:
        cmap='seismic'
    plt.axis('off')
    cax = ax.imshow(input, interpolation='nearest', cmap=cmap ,norm=MidpointNormalize(midpoint=0))

    if(x_axis !=None):
        fig.text(0.5, 0.01, x_axis, ha='center' , fontsize=14)
    
    if(y_axis !=None):
        fig.text(0.05, 0.5, y_axis, va='center', rotation='vertical', fontsize=14)
    fig.tight_layout()
    ax.set_title(title)
    plt.savefig(Loc_Graph+saveLocation+'.png' )


    if(show):
        plt.show()


def main(args):

    TestingLabel = Helper.load_CSV(args.data_dir+"SimulatedTestingLabels"+args.DataName+"_F"+str(args.input_size)+"_TS_"+str(args.sequence_length)+"_IMPSTART_"+str(args.importance)+".csv")
    TestingType=TestingLabel[:,-1]
    Types = np.unique(TestingType)
    TestingLabel= TestingLabel[:,0]
    TestingLabel=TestingLabel.reshape(TestingLabel.shape[0],)
    TestingLabel = Helper.reOrderLabels(TestingLabel.tolist())
    TestingLabel=np.array(TestingLabel)
    ModelTypes=['LSTM','InputCellAttention']
    saliencies=[]
    for model in ModelTypes:
        saliency= Helper.load_CSV(Results+args.DataName+"_"+model+"_saliencies.csv")
        saliency=saliency.reshape(saliency.shape[0] , args.sequence_length,args.input_size)
        saliencies.append(saliency)

    for s , saliency in enumerate(saliencies):
        for j in range(Types.shape[0]):
            posGrad=[]
            negGrad=[]
            for i in range (saliency.shape[0]):
                if(TestingLabel[i]==0):
                    negGrad.append(saliency[i])
                else:
                    posGrad.append(saliency[i])

            meanPosGrad = np.mean(np.array(posGrad), axis=0)
            meanNegGrad = np.mean(np.array(negGrad), axis=0)
 
            plotHeatMapExampleWise(meanPosGrad,ModelTypes[s]+" Postive Sample Saliency " ,ModelTypes[s]+"posGradMeanMat",greyScale=True,flip=True,x_axis='Time',y_axis='Saliency')

            plotHeatMapExampleWise(meanNegGrad,ModelTypes[s]+" Negative Sample Saliency " ,ModelTypes[s]+"negGradMeanMat",greyScale=True,flip=True,x_axis='Time',y_axis='Saliency')


  
def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--DataName', type=str ,default="TopBox")
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--input_size', type=int,default=100)
    parser.add_argument('--importance', type=str, default=0)
    parser.add_argument('--data-dir', help='Data  directory', action='store', type=str ,default="../Data/")
    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
