'''
This is the main function.
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
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold
import sys
import numpy.random
from sklearn import preprocessing
# from time import time

# print (PYTHONPATH)




def single_fold(k, topk, dataset,datasetnum, kernel, cmin,cmax,number_of_cs, skfseed, percent_of_priv, percentageofinstances,take_top_t):

        if take_top_t not in ['top','bottom']:
                print('take top t should be "top"or "bottom"')
                sys.exit()

        print('using  {}% of training data instances'.format(percentageofinstances))
        print('percentage of discarded info used as priv:{}'.format(percent_of_priv))
        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        print('cvalues',c_values)


        print('word',take_top_t)
        output_directory = get_full_path(('Desktop/Privileged_Data/Save_For_Ollie/tech'.format(datasetnum,)))
        print (output_directory)

        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")

        cross_validation_folder = os.path.join(output_directory,'{}-{}'.format(skfseed,k))
        try:
            os.makedirs(cross_validation_folder)
        except OSError:
            if not os.path.isdir(cross_validation_folder):
                raise

        all_training, all_testing, training_labels, testing_labels,train_indices, test_indices = get_train_and_test_this_fold(dataset,datasetnum,k,skfseed)
        print('train indices',train_indices)
        print('test indices', train_indices)

        print (all_training.shape)


        n_top_feats = topk

        # n_top_feats = topk*all_training.shape[1]//100
        print ('n top feats',n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))

        # start_time=time()
        ########## GET BEST C FOR RFE

        best_rfe_param = get_best_RFE_C(all_training, training_labels, c_values, n_top_feats, stepsize,
                                         datasetnum, topk)
        print('best rfe param', best_rfe_param)

        ###########  CARRY OUT RFE, GET ACCURACY

        svc = SVC(C=best_rfe_param, kernel=kernel, random_state=k)
        rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=stepsize)
        print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        print (all_testing.shape,testing_labels.shape)
        print ('num of chosen feats',sum(x == 1 for x in rfe.support_))

        best_n_mask = rfe.support_
        normal_features_training = all_training[:,best_n_mask].copy()
        normal_features_testing = all_testing[:,best_n_mask].copy()
        privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()

        svc = SVC(C=best_rfe_param, kernel=kernel, random_state=k)
        svc.fit(normal_features_training,training_labels)
        rfe_accuracy = svc.score(normal_features_testing,testing_labels)
        print ('rfe accuracy (using slice):',rfe_accuracy)


        with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,topk)),'a') as cv_svm_file:
            cv_svm_file.write(str(rfe_accuracy)+",")

        print('normal train shape {},priv train shape {}'.format(normal_features_training.shape,privileged_features_training.shape))
        print('normal testing shape {}'.format(normal_features_testing.shape))
        np.save(os.path.join(output_directory,'tech{}-{}-{}-train_normal'.format(datasetnum,skfseed,k)),normal_features_training)
        np.save(os.path.join(output_directory,'tech{}-{}-{}-train_priv'.format(datasetnum,skfseed,k)),privileged_features_training)
        np.save(os.path.join(output_directory,'tech{}-{}-{}-test_normal'.format(datasetnum,skfseed,k)),normal_features_testing)
        np.save(os.path.join(output_directory,'tech{}-{}-{}-train_labels'.format(datasetnum,skfseed,k)),training_labels)
        np.save(os.path.join(output_directory,'tech{}-{}-{}-test_labels'.format(datasetnum,skfseed,k)),testing_labels)





#[254 174 219 197 137]

for k in range(1,10):
    for datasetnum in [254,174, 219, 197, 137]:
        print(single_fold(k=k, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=0, percent_of_priv=100, percentageofinstances=100,take_top_t='top'))