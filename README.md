# Input-Cell Attention
Code implementing architecture introduced in "Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks" by
Aya Abdelsalam Ismail, Mohamed Gunady, Luiz Pessoa, Hector Corrada Bravo*, Soheil Feizi*.

![alt text](Images/cellAttentionLstm.png)


## Creating datasets:
1. createSimulationData.py --> Creates TopBox,MiddleBox,BottomBox,ThreeUpperBoxes,ThreeMiddleBoxes datasets to choose which dataset to create uncomment it in the code
2. createMixedSimulationData.py --> Creates MixedBoxes Dataset.
3. createMovingSimulationData.py --> Creates datasets for moving boxes experiments
Datasets will be saved in Data folder
## Train Models:
- Uses trainLSTMModels.py
- Uncomment the model you would like to train and write the dataset details in the arguments
- Models will be in Models folder
## Create Saliency:
- Uses vanillaSaliencyClean.py
- Uncomment the model you would like to train and write the dataset details in the arguments'
- Saliencies and gradients will be saved in Results folder
## Get Statics:
-  To get table used in the paper uses BoxStat.py it reads from Salincy from Results folder. (uncomment the desired models)
-  To get plot for moving boxes:
    1. Run MovingBoxStat.py
    2. Run MovingBoxPlot.py
