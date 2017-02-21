
import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold
import sys
import os
import numpy as np

datasetnum=254
skfseed=0
k=0
output_directory = get_full_path(('Desktop/Privileged_Data/Save_For_Ollie/tech'))

data = np.load(os.path.join(output_directory,'tech{}-{}-{}-train_normal.npy'.format(datasetnum,skfseed,k)))
testdata = np.load(os.path.join(output_directory,'tech{}-{}-{}-test_normal.npy'.format(datasetnum,skfseed,k)))
priv = np.load(os.path.join(output_directory,'tech{}-{}-{}-train_priv.npy'.format(datasetnum,skfseed,k)))
print(data.shape,testdata.shape,priv.shape)
