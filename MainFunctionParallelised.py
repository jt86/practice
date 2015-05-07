__author__ = 'jt306'

import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import f1_score, pairwise
from sklearn import svm, linear_model
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from SVMplus3 import svmplusQP, svmplusQP_Predict
import sklearn.preprocessing as preprocessing
from Get_Figures import get_figures
from ParamEstimation2 import param_estimation
from FeatSelection import univariate_selection, recursive_elimination
import time
from FeatSelection import get_ranked_indices, recursive_elimination2
from InitialFeatSelection import get_best_feats
from Get_Mean import get_mean_from, get_error_from
import pdb
from BringFoldsTogether import bring_folds_together
from SingleFold import single_fold

def main_function(output_directory, num_folds,
                  cmin, cmax,number_of_cs, peeking, dataset, rank_metric, bottom_n_percent=0,
                logger=None, cstarmin=None, cstarmax=None, kernel=None, take_t=False):



    for k in range (num_folds):
        single_fold(k, num_folds, take_t, bottom_n_percent,
        rank_metric, dataset, logger, peeking, kernel,cmin,cmax,number_of_cs, cstarmin, cstarmax, output_directory)


    # if take_t == True:
    #     x_axis_list = list_of_percentages
    # else:
    #     x_axis_list = numbers_of_features_list
