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

from sklearn.preprocessing import MinMaxScaler


sampleLength=50

def rescale_score_by_abs (score, max_score, min_score):
    """
    rescale positive score to the range [0.5, 1.0], negative score to the range [0.0, 0.5],
    using the extremal scores max_score and min_score for normalization
    """
    
    # CASE 1: positive AND negative scores occur --------------------
    if max_score>0 and min_score<0:
    
        if max_score >= abs(min_score):   # deepest color is positive
            if score>=0:
                return 0.5 + 0.5*(score/max_score)
            else:
                return 0.5 - 0.5*(abs(score)/max_score)

        else:                             # deepest color is negative
            if score>=0:
                return 0.5 + 0.5*(score/abs(min_score))
            else:
                return 0.5 - 0.5*(score/min_score)   
    
    # CASE 2: ONLY positive scores occur -----------------------------       
    elif max_score>0 and min_score>=0: 
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5*(score/max_score)
    
    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score<=0 and min_score<0: 
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5*(score/min_score)   


def normalizeAtoB(score,max,min,a=0,b=1):
    newscore= (b-a)*((score-min)/(max-min))+a
    return newscore
def rescale(image,changetodiscrete=True):
    newImage=np.zeros((image.shape))
    max_score, min_score = np.amax(image), np.amin(image)
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            score= normalizeAtoB (image[i,j], max_score, min_score)


            if(changetodiscrete):
                if(score>=2.5):
                    newImage[i,j]=1
                elif(score<=-2.5):
                    newImage[i,j]=-1
            else:
                newImage[i,j]=score
    return newImage




def plotHeatMapMean(input,title,saveLocation ,  ):
	# input=np.mean(input, axis=0)
	max,min=np.max(input) , np.min(input)
	newInput=[]
	
	for score in input:
		newInput.append(rescale_score_by_abs (score, max, min))
	newInput = np.array(newInput)
	newInput = newInput.reshape(sampleLength,500)
	fig, ax = plt.subplots()

	cax = ax.imshow(newInput, interpolation='nearest', cmap='seismic' ,  vmin =0,vmax=1 )
	ax.set_title(title)
	cbar = fig.colorbar(cax)
	plt.savefig(Loc_Graph+saveLocation+'.png',box_inches='tight' ,  pad_inches = 0)
	# plt.show()



# def plotHeatMapExampleWise(input,title, saveLocation,reshape=False , max=None,min=None,greyScale=False,flip=False,x_axis=None,y_axis=None,show=True):
    
#     if(reshape):
#         input = input.reshape(50,500)

#     for i in range(input.shape[0]):
#         for j in range(input.shape[1]):
#             input[i,j] = normalizeAtoB(input[i,j] ,max,min)
            
#     if(flip):
#         input=np.transpose(input)
#     fig, ax = plt.subplots()
#     # cax = ax.imshow(input, interpolation='nearest', cmap='seismic' ,  vmin =0,vmax=1 )
#     # 
#     # plt.savefig(Loc_Graph+saveLocation+'.png',box_inches='tight' ,  pad_inches = 0)
#     if(greyScale):
#         cmap='gray'
#     else:
#         cmap='seismic'
#     if(max==None and min==None):
#         cax = ax.imshow(input, interpolation='nearest', cmap=cmap  )
#     else:
#         cax = ax.imshow(input, interpolation='nearest', cmap=cmap ,  vmin =0,vmax=1 )
#     # cbar = fig.colorbar(cax)

#     ax.set_title(title)
#     if(x_axis !=None):
#         fig.text(0.5, 0.01, x_axis, ha='center' , fontsize=30)
    
#     if(y_axis !=None):
#         fig.text(0.09, 0.5, y_axis, va='center', rotation='vertical', fontsize=30)
#     fig.tight_layout()
#     plt.axis('off')
#     plt.autoscale(tight=True)

#     plt.savefig(Loc_Graph+saveLocation+'.png' , box_inches='tight' ,  pad_inches = 0)

#     # plt.savefig(Loc_Graph+saveLocation+'.png' , box_inches='tight' ,  pad_inches = 0)
#     # plt.imsave(saveLocation+'.png', input, cmap='gray', format="png")
#     if(show):
#         plt.show()

def plotHeatMapExampleWise(input,title, saveLocation,reshape=False , max=None,min=None,greyScale=False,flip=False,x_axis=None,y_axis=None,show=True):
    
    if(reshape):
        input = input.reshape(50,500)

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            input[i,j] = normalizeAtoB(input[i,j] ,max,min)
            
    if(flip):
        input=np.transpose(input)
    fig, ax = plt.subplots()

    if(greyScale):
        cmap='gray'
    else:
        cmap='seismic'
    plt.axis('off')
    if(max==None and min==None):
        cax = ax.imshow(input, interpolation='nearest', cmap=cmap  )
    else:
        cax = ax.imshow(input, interpolation='nearest', cmap=cmap ,  vmin =0,vmax=1 )
    cbar = fig.colorbar(cax)

    # ax.set_title(title)
    # if(x_axis !=None):
    #     fig.text(0.5, 0.01, x_axis, ha='center' , fontsize=20)
    
    # if(y_axis !=None):
    #     fig.text(0.05, 0.5, y_axis, va='center', rotation='vertical', fontsize=20)
    fig.tight_layout()
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.subplots_adjust(left=0., right=0.95, top=0.95, bottom=0.05)
    # plt.subplots_adjust(left=1, right=2, top=1, bottom=0)


    plt.savefig(Loc_Graph+saveLocation+'.png' )

    # plt.savefig(Loc_Graph+saveLocation+'.png' , box_inches='tight' ,  pad_inches = 0)
    # plt.imsave(saveLocation+'.png', input, cmap='gray', format="png")
    if(show):
        plt.show()


def plotfeatureValueMean(input,title,saveLocation):
	input_MEAN=np.mean(input, axis=0)
	input_MEAN = input_MEAN.reshape(sampleLength,500)
	std= np.std(input_MEAN, axis=0)
	# print(input_MEAN.shape,std.shape)
	input_MEAN=np.mean(input_MEAN, axis=0)
	# print(input_MEAN.shape,std.shape ,np.max(input_MEAN) , np.min(input_MEAN))
	plt.figure()
	plt.title(title)
	plt.barh(range(input_MEAN.shape[0]), input_MEAN,
	       color="r", align="center")
	plt.savefig(Loc_Graph+saveLocation+'.png')




def normalizeAtoB(score,max,min,a=0,b=1):
    newscore= (b-a)*((score-min)/(max-min))+a
    return newscore



def plotDataHeatMap(input,title,saveLocation,greyScale=False):
    newInput=np.zeros(input.shape)
    max,min=np.max(input) , np.min(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            newInput[i,j] = rescale_score_by_abs (input[i,j], max, min)
    fig, ax = plt.subplots()

    cax = ax.imshow(newInput, interpolation='nearest', cmap='seismic' ,  vmin =0,vmax=1 )
    if(greyScale):
        cax = ax.imshow(newInput, interpolation='nearest', cmap='gray' ,  vmin =0,vmax=1 )

    ax.set_title(title)
    cbar = fig.colorbar(cax)
    plt.savefig(Loc_Graph+saveLocation+'.png',box_inches='tight' ,  pad_inches = 0)
    plt.show()

def linePlot(input , title,saveLocation):

    numFeatures=input.shape[1]
    numTimesteps=input.shape[0]
    

    colors = plt.cm.jet(np.linspace(0,1,numFeatures))

    for i in range(numFeatures):
        plt.plot(input[:,i], color=colors[i])
    plt.title(title)
    plt.savefig(Loc_Graph+saveLocation+'.png',box_inches='tight' ,  pad_inches = 0)
    # set_xlim([0, numTimesteps])
    plt.show()

# def plotHeatMapExampleWise(inputs, title, saveLocation ):
# 	print("STARTING plotHeatMapExampleWise")
# 	newInputs=[]
# 	for i  in range(inputs.shape[0]):
# 		input=inputs[i,:]
# 		max,min=np.max(input) , np.min(input)
# 		newInput=[]
# 		for score in input:
# 			newInput.append(rescale_score_by_abs (score, max, min))
# 		newInput = np.array(newInput)
# 		newInput = newInput.reshape(sampleLength,500)
# 		newInputs.append(newInput)

# 	fig, axes = plt.subplots(nrows=10, ncols=1)
# 	print(len(newInputs))
# 	nrow=0
# 	ncol=0
# 	for num,newInput in enumerate(newInputs):
# 		if(num==10):
# 			break
# 		print(num)
# 		im = axes[num].imshow(newInputs[num], interpolation='nearest', cmap='seismic' ,  vmin =0,vmax=1 )
# 		axes[num].set_title(str(num))
# 	fig.subplots_adjust(right=0.8)
# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# 	fig.colorbar(im, cax=cbar_ax)
# 	fig.suptitle(title, fontsize=14)
# 	plt.savefig(Loc_Graph+saveLocation+'.png')
	# plt.show()


# def plotHeatMapExampleWise(inputs, title, saveLocation ):
# 	newInputs=[]
# 	for i  in range(inputs.shape[0]):
# 		input=inputs[i,:]
# 		max,min=np.max(input) , np.min(input)
# 		print(input.shape)
# 		newInput=[]
# 		for score in input:
# 			newInput.append(rescale_score_by_abs (score, max, min))
# 		newInput = np.array(newInput)
# 		newInput = newInput.reshape(sampleLength,500)
# 		newInputs.append(newInput)

# 	fig, axes = plt.subplots(nrows=3, ncols=2)
# 	nrow=0
# 	ncol=0
# 	for num,newInput in enumerate(newInputs):
# 		im = axes[nrow,ncol].imshow(newInputs[num], interpolation='nearest', cmap='seismic' ,  vmin =0,vmax=1 )
# 		axes[nrow,ncol].set_title(ModelColumns[num])
# 		nrow+=1
# 		if(nrow>=3):
# 			nrow=0
# 			ncol+=1
# 	fig.subplots_adjust(right=0.8)
# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# 	fig.colorbar(im, cax=cbar_ax)
# 	fig.suptitle(title, fontsize=14)
# 	plt.savefig(Loc_Graph+saveLocation+'.png')
# 	# plt.show()






