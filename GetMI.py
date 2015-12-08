import os
import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE, chi2, f_classif
from sklearn.svm import SVC
# from GetSingleFoldData import get_train_and_test_this_fold
from sklearn import preprocessing
import sys
# from GetFeatSelectionData import get_train_and_test_this_fold
from GetSingleFoldData import get_train_and_test_this_fold
from sklearn.metrics import normalized_mutual_info_score as mi

def single_fold(k, topk, dataset,datasetnum, kernel, cmin,cmax,number_of_cs, skfseed, percent_of_priv=100):

        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        print('cvalues',c_values)

        output_directory = get_full_path(('Desktop/Privileged_Data/GetScore-{}-{}to{}-{}-{}').format(dataset,cmin,cmax,stepsize,topk))

        print (output_directory)
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")


        all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset,datasetnum,k,skfseed)
        all_features = np.vstack((all_training, all_testing))
        all_labels = np.hstack((training_labels,testing_labels))

        n_top_feats = topk

        print ('n top feats',n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))


        ########## GET BEST C FOR RFE

        best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize,output_directory,datasetnum,topk)
        print('best rfe param', best_rfe_param)

        ###########  CARRY OUT RFE, GET ACCURACY

        svc = SVC(C=best_rfe_param, kernel="linear", random_state=k)
        rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=stepsize)
        print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        print (all_testing.shape,testing_labels.shape)
        print ('num of chosen feats',sum(x == 1 for x in rfe.support_))

        best_n_mask = rfe.support_
        priv_feats_all = all_features[:,np.invert(rfe.support_)].copy()
        normal_feats_all = all_features[:,best_n_mask].copy()

        priv_feats_train = all_training[:,np.invert(rfe.support_)].copy()
        normal_feats_train = all_training[:,best_n_mask].copy()

        priv_feats_test = all_testing[:,np.invert(rfe.support_)].copy()
        normal_feats_test = all_testing[:,best_n_mask].copy()

        ########### WRITE SCORES FOR EACH FEATURE TO FILE

        with open(os.path.join(output_directory,'normal-all-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as normal_chi2_file:
                for item in (f_classif(normal_feats_all, all_labels)[0]):
                        normal_chi2_file.write(str(item)+',')
        with open(os.path.join(output_directory,'priv-all-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as priv_chi2_file:
                for item in (f_classif(priv_feats_all, all_labels)[0]):
                        priv_chi2_file.write(str(item)+',')

        with open(os.path.join(output_directory,'normal-train-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as normal_chi2_file:
                for item in (f_classif(normal_feats_train, training_labels)[0]):
                        normal_chi2_file.write(str(item)+',')
        with open(os.path.join(output_directory,'priv-train-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as priv_chi2_file:
                for item in (f_classif(priv_feats_train, training_labels)[0]):
                        priv_chi2_file.write(str(item)+',')

        with open(os.path.join(output_directory,'normal-test-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as normal_chi2_file:
                for item in (f_classif(normal_feats_test, testing_labels)[0]):
                        normal_chi2_file.write(str(item)+',')
        with open(os.path.join(output_directory,'priv-test-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as priv_chi2_file:
                for item in (f_classif(priv_feats_test, testing_labels)[0]):
                        priv_chi2_file.write(str(item)+',')


def get_random_array(num_instances,num_feats):
    random_array = np.random.rand(num_instances,num_feats)
    random_array = preprocessing.scale(random_array)
    return random_array


# single_fold(k=5, topk=300, dataset='tech', datasetnum=0, kernel='linear', cmin=3, cmax=3, number_of_cs=1,skfseed=1, percent_of_priv=100)#,stepsize=0.6)

