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
from GetRSingleFoldData import get_techtc_data



def save_instance_and_feature_indices_for_R(k, topk, dataset,datasetnum, kernel, cmin,cmax,number_of_cs, skfseed):

        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)

        all_training, all_testing, training_labels, testing_labels,train_indices,test_indices = get_train_and_test_this_fold(dataset,datasetnum,k,skfseed)

        ########## GET BEST C FOR RFE
        best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, topk,stepsize,datasetnum,topk)
        print('best rfe param', best_rfe_param)
        ###########  CARRY OUT RFE, GET ACCURACY
        svc = SVC(C=best_rfe_param, kernel=kernel, random_state=k)
        rfe = RFE(estimator=svc, n_features_to_select=topk, step=stepsize)
        print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        print (all_testing.shape,testing_labels.shape)

        selected_feat_indices = np.where(rfe.support_ == True)[0]
        unselected_feat_indices = np.where(rfe.support_ == False)[0]
        print ('selected',selected_feat_indices.shape)
        print('unselected', unselected_feat_indices.shape)

        # save_string = get_full_path('/Volumes/LocalDataHD/j/jt/jt306/Documents/CVPR2016_Rcode/saved_indices/top{}-{}{}-{}-{}-'.format(topk,dataset,datasetnum,k,skfseed))
        save_string = get_full_path('Desktop/Rcode/saved_indices/top{}-{}{}-{}-{}-'.format(topk, dataset,datasetnum,k,skfseed))

        np.savetxt(save_string+'train_instances_indices',train_indices)
        np.savetxt(save_string + 'test_instances_indices',test_indices)
        np.savetxt(save_string + 'selected_feat_indices',selected_feat_indices)
        np.savetxt(save_string + 'unselected_feat_indices', unselected_feat_indices)



# save_instance_and_feature_indices_for_R(k=3, topk=500, dataset='tech', datasetnum=40, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=4, percent_of_priv=100, percentageofinstances=100)#, take_top_t='bottom')


def save_dataset_for_R(datasetnum):

        class0_data,class1_data = get_techtc_data(datasetnum)
        class0_labels = [float(-1)] * class0_data.shape[0]
        class1_labels = [float(1)] * class1_data.shape[0]
        all_labels = (np.r_[class0_labels, class1_labels])
        print(all_labels.shape)

        all_data = np.vstack([class0_data, class1_data])
        print (all_data.shape)
        print (all_labels.shape)

        save_string = get_full_path('Desktop/Rcode/saved_datasets/tech{}'.format(datasetnum))
        np.savetxt(save_string + 'data', all_data)
        np.savetxt(save_string + 'labels', all_labels)

        print(type(all_data[0][0]))
        print(type(all_labels[0]))


# save_dataset_for_R(39)

#

#
# data = np.load(get_full_path('/Volumes/LocalDataHD/j/jt/jt306/Documents/CVPR2016_Rcode/saved_datasets/tech39data.npy'))
#
# print (data[0,0:130])
# labels = np.load(get_full_path('/Volumes/LocalDataHD/j/jt/jt306/Documents/CVPR2016_Rcode/saved_datasets/tech39labels.npy'))
#
# topk=500;dataset='tech';datasetnum=39;k=3;skfseed=4
#
#
# saved_training_indices = [int(item) for item in np.loadtxt(get_full_path('/Volumes/LocalDataHD/j/jt/jt306/Documents/CVPR2016_Rcode/saved_indices/top{}-{}{}-{}-{}-train_instances_indices'.format(topk,dataset,datasetnum,k,skfseed)))]
# print (len(saved_training_indices))
#
# saved_testing_indices = [int(item) for item in np.loadtxt(get_full_path('/Volumes/LocalDataHD/j/jt/jt306/Documents/CVPR2016_Rcode/saved_indices/top{}-{}{}-{}-{}-test_instances_indices'.format(topk,dataset,datasetnum,k,skfseed)))]
# print (len(saved_testing_indices))
#
# train_items = data[saved_training_indices,:]
# print (train_items.shape)
#
# test_items = data[saved_testing_indices,:]
# print (test_items.shape)