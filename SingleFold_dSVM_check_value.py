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




def single_fold(k, top_k, dataset, datasetnum, kernel, cmin, cmax, number_of_cs, skfseed, percent_of_priv, percentageofinstances, take_top_t):



        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        # print('cvalues',c_values)

        output_directory = get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE')
        # print (output_directory)
        #
        #
        # with open(os.path.join(output_directory, 'tech{}-dScore.csv'),'a') as savefile:
        #     savefile.write('hello,hello')

        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")

        cross_validation_folder = os.path.join(output_directory,'cross-validation{}'.format(skfseed))
        try:
            os.makedirs(cross_validation_folder)
        except OSError:
            if not os.path.isdir(cross_validation_folder):
                raise

        all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset,datasetnum,k,skfseed)


        param_estimation_file.write("\n\n n={},fold={}".format(top_k, k))


        # all_training = all_training[:,:1000]
        # all_testing = all_testing[:, :1000]


        ########## GET BEST C FOR RFE

        # best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize,cross_validation_folder,datasetnum,topk)
        best_rfe_param = get_best_RFE_C(all_training, training_labels, c_values, top_k, stepsize,
                                        datasetnum, top_k)
        # print('best rfe param', best_rfe_param)

        ###########  CARRY OUT RFE, GET ACCURACY

        svc = SVC(C=best_rfe_param, kernel=kernel, random_state=k)
        rfe = RFE(estimator=svc, n_features_to_select=top_k, step=stepsize)
        # print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        # print (all_testing.shape,testing_labels.shape)
        # print ('num of chosen feats',sum(x == 1 for x in rfe.support_))


        normal_features_training = all_training[:,rfe.support_].copy()
        normal_features_testing = all_testing[:,rfe.support_].copy()
        privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()


        ######### Fit SVM to just the privileged features and then take the slacks

        c = get_best_C(privileged_features_training, training_labels, c_values, cross_validation_folder, datasetnum, top_k)
        svc = SVC(C=c, kernel=kernel, random_state=k)
        svc.fit(privileged_features_training,training_labels)
        # rfe_accuracy = svc.score(normal_features_testing,testing_labels)
        # print ('rfe accuracy (using slice):',rfe_accuracy
        # print('coefficient:',svc.coef_)
        # print('coefficient:', svc.coef_.shape)
        #
        # print('intercept:',svc.intercept_)
        # print ('decision function:\n',svc.decision_function(privileged_features_training))
        # print ('\nslacks\n')


        # save decion functions for SVC trained on privileged features only. Dec function = (coefficients*values) + bias
        decision_functions = svc.decision_function(privileged_features_training)

        # d_i = 1 - y *(privileged value)

        d_i = np.array([1 - (training_labels[i] * svc.decision_function(privileged_features_training)[i])  for i in range(len(training_labels))])
        print (d_i.shape)
        d_i = np.reshape(d_i, (d_i.shape[0], 1))

        # print (training_labels)
        # print (svc.decision_function(privileged_features_training))
        # print (training_labels * svc.decision_function(privileged_features_training))
        #
        # print('best c rfe', best_rfe_param)
        # print('best c', c)
        # print('di',d_i)

        with open(os.path.join(output_directory, 'tech{}-Cparams.csv'.format(datasetnum)),'a') as savefile:
            savefile.write('rfe'+str(best_rfe_param)+'\n')
            savefile.write('svm on unselected'+str(c)+'\n')
            savefile.write('svm on unselected' + str(c) + '\n')
            savefile.write('svm on unselected' + str(c) + '\n')



        # with open(os.path.join(output_directory, 'tech{}-dScore.csv'.format(datasetnum)),'a') as savefile:
        #     savefile.write(str(best_rfe_param)+'\n')
        #     savefile.write(str(c)+'\n')
        #     for item in d_i:
        #         savefile.write(str(item).strip('[]')+',')
        #     savefile.write('\n')
        #     savefile.write('mean,'+str(np.mean(d_i))+'\n')
        #     savefile.write('max,' + str(np.max(d_i))+'\n')
        #     savefile.write('min,' + str(np.min(d_i))+'\n')
        #     savefile.write('stdev,' + str(np.std(d_i))+'\n')

        return(d_i)

mins,maxes,means,stdevs=[],[],[],[]
dataset='tech'
for datasetnum in range (295): #5
    d_i=single_fold(k=1, top_k=300, dataset='tech', datasetnum=datasetnum, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='bottom')
    print ('di',d_i)
    print(np.min(d_i))
    mins.append(np.min(d_i))
    maxes.append(np.max(d_i))
    means.append(np.mean(d_i))
    stdevs.append(np.std(d_i))
print ('maxes',maxes)
print ('mins',mins)
print ('means',means)
print ('stdevs',stdevs)


np.save(os.path.join(get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE/maxes')),np.array(maxes))
np.save(os.path.join(get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE/mins')),np.array(mins))
np.save(os.path.join(get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE/means')),np.array(means))
np.save(os.path.join(get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE/stdevs')),np.array(stdevs))
