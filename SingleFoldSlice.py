import os
import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold

from sklearn import preprocessing
import sys

import numpy.random

def single_fold(k, topk, dataset,datasetnum, kernel, cmin,cmax,number_of_cs, skfseed, percent_of_priv, percentageofinstances):

        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        print('cvalues',c_values)

        # outer_directory = get_full_path(('Desktop/Privileged_Data/10x10-{}-ALLCV-{}to{}-featsscaled-bottom{}-{}/').format(dataset,cmin,cmax,percent_of_priv,topk))
        # output_directory = os.path.join(get_full_path(outer_directory),'fixedCandCstar-10fold-{}-{}-RFE-baseline-step={}-percent_of_priv={}'.format(dataset,datasetnum,stepsize,percent_of_priv))

        output_directory = get_full_path(('Desktop/Privileged_Data/10x10-{}-ALLCV{}to{}-featsscaled-step{}/tech{}/top{}chosen-{}percentinstances/').format(dataset,cmin,cmax,stepsize,datasetnum,topk,percentageofinstances))

        print (output_directory)
        sys.exit()
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
        orig_num_train_instances = all_training.shape[0]
        num_of_train_instances = orig_num_train_instances*percentageofinstances//100
        indices = np.random.choice(orig_num_train_instances,num_of_train_instances,replace=False)
        all_training = all_training.copy()[indices,:]
        training_labels = training_labels[indices]
        print (all_training.shape)
        print (training_labels.shape)
        print(indices)
        # sys.exit()


        # n_top_feats = topk

        n_top_feats = topk*all_training.shape[1]//100
        print ('n top feats',n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))


        ########## GET BEST C FOR RFE

        best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize,cross_validation_folder,datasetnum,topk)
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
        ##############################  BASELINE - all features
        # if topk==75:
        best_C_baseline = get_best_C(all_training, training_labels, c_values, cross_validation_folder,datasetnum,topk)
        # best_C_baseline=best_rfe_param
        print('all feats best c',best_C_baseline)

        print ('all training shape',all_training.shape)

        clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=k)
        clf.fit(all_training, training_labels)
        baseline_predictions = clf.predict(all_testing)
        print ('baseline',accuracy_score(testing_labels,baseline_predictions))

        with open(os.path.join(cross_validation_folder,'baseline-{}.csv'.format(k)),'a') as baseline_file:
            baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')

        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING



        ##### THIS PART TO GET A SUBSET OF PRIV INFO
        # print('privileged',privileged_features_training.shape)
        # all_features_ranking = rfe.ranking_[np.invert(best_n_mask)]
        # privileged_features_training = privileged_features_training[:,np.argsort(all_features_ranking)]
        # num_of_priv_feats=percent_of_priv*privileged_features_training.shape[1]//100
        #
        #
        # privileged_features_training = privileged_features_training[:,-num_of_priv_feats:]
        # print ('privileged data shape',privileged_features_training.shape)

        # privileged_features_training = get_random_array(privileged_features_training.shape[0],privileged_features_training.shape[1]*5)
        # print ('random data size',privileged_features_training.shape)


        # c_star_values = [10., 5., 2., 1., 0.5, 0.2, 0.1]
        # c_star_values=[0.0001, 0.001, 0.01, 0.1]
        c_star_values=c_values
        # c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training,
        #                                 c_svm_plus, c_star_values,cross_validation_folder,datasetnum, topk)

        c_svm_plus,c_star_svm_plus = get_best_CandCstar(normal_features_training,training_labels, privileged_features_training,
                                         c_values, c_star_values,cross_validation_folder,datasetnum, topk)

        duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)
        lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()

        accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))
        print('svm+ accuracy',(accuracy_lupi))
        with open(os.path.join(cross_validation_folder,'lupi-{}-{}.csv'.format(k,topk)),'a') as cv_lupi_file:
            cv_lupi_file.write(str(accuracy_lupi)+',')
        print ('k=',k, 'seed=',skfseed,'topk',topk)
        return (rfe_accuracy,accuracy_lupi )


def get_random_array(num_instances,num_feats):
    random_array = np.random.rand(num_instances,num_feats)
    random_array = preprocessing.scale(random_array)
    return random_array

# value = 1
# print(single_fold(k=4, topk=300, dataset='tech', datasetnum=0, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=7, percent_of_priv=100, percentageofinstances=50))
#  single_fold(k=1, topk=5, dataset='arcene', datasetnum=0, kernel='linear', cmin=value, cmax=value, number_of_cs=1,skfseed=9, percent_of_priv=100,percentage_of_instances=50)