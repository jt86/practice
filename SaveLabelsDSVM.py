'''
Used to train a standard SVM and save the slack variable for use with dSVM+
Things to check before running: (1) values of C, (2) output directory and whether old output is there
(3) number of jobs in go-practice-submit.sh matches desired number of settings to run in Run Experiment
(4) that there is no code test run
(5) data is regularised as desired in GetSingleFoldData
(6) params including number of folds and stepsize set correctly
'''

import os

# print os.environ['HOME']

import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold
import sys
import numpy.random
from sklearn import preprocessing
# from time import time

# print (PYTHONPATH)




def save_train_labels(k, dataset, datasetnum, skfseed):
        np.random.seed(k)
        all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset,datasetnum,k,skfseed)
        np.save(get_full_path('Desktop/SavedTrainLabels/{}-{}-{}').format(datasetnum,skfseed,k),training_labels)

for datasetnum in range (295):
    for skfseed in range(10):
        for k in range(10):
            save_train_labels(k,'tech',datasetnum,skfseed)