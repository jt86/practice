from __future__ import division
import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import pairwise, accuracy_score
from sklearn import svm, linear_model
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import param_estimation,get_best_Cstar, get_best_C, get_best_RFE_C

from Get_Full_Path import get_full_path
from Get_Awa_Data import get_awa_data
from CollectBestParams import collect_best_rfe_param
from sklearn.feature_selection import RFE
from FromViktoriia import getdata
import numpy
from sklearn.svm import SVC, LinearSVC
from GetSingleFoldData import get_train_and_test_this_fold
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

def single_fold(k, percentage, dataset, kernel, cmin,cmax,number_of_cs,rank_metric=chi2):

        np.random.seed(k)

        c_values = np.logspace(cmin,cmax,number_of_cs)
        print 'cvalues',c_values

        outer_directory = get_full_path('Desktop/Privileged_Data/')
        output_directory = os.path.join(get_full_path(outer_directory),'{}-chi2'.format(dataset))
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise

        # original_features_array, labels_array, tuple = get_feats_and_labels(dataset)
        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
        # chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")




        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        try:
            os.makedirs(cross_validation_folder)
        except OSError:
            if not os.path.isdir(cross_validation_folder):
                raise


        if 'awa' in dataset:
            class_id =dataset[-1]
            all_training, all_testing, training_labels, testing_labels = get_awa_data("", class_id)
        else:
            all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset)


        ordered_feats=univariate_selection(all_training,training_labels,rank_metric)
        sorted_training = all_training[:, ordered_feats]
        sorted_testing = all_testing[:, ordered_feats]



        total_number_of_feats = all_training.shape[1]
        list_of_percentages = [5,10,25,50,75]


        topK = percentage/100
        n_top_feats=int(topK*total_number_of_feats)

        normal_features_training = sorted_training[:,:n_top_feats]
        normal_features_testing = sorted_testing[:,:n_top_feats]
        privileged_features_training=sorted_training[:,:n_top_feats]


        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))
        ############


        # method = 'privfeat_rfe_top'
        CV_best_param_folder = os.path.join(output_directory,'{}CV/'.format(dataset))
        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        try:
            os.makedirs(CV_best_param_folder)
        except OSError:
            if not os.path.isdir(CV_best_param_folder):
                raise



        best_C_SVM = get_best_C(normal_features_training,training_labels,c_values)
        with open(os.path.join(cross_validation_folder,'best_svm_param{}.txt'.format(k)),'a') as best_params_doc:
            best_params_doc.write("\n"+str(best_C_SVM))
        print 'best rfe param', best_C_SVM
        clf = svm.SVC(C=best_C_SVM, kernel=kernel,random_state=1)
        clf.fit(normal_features_training, training_labels)
        svm_predictions = clf.predict(normal_features_testing)
        with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,percentage)),'a') as cv_svm_file:
            cv_svm_file.write(str(accuracy_score(testing_labels,svm_predictions))+',')

        # ##############################  BASELINE - all features
        best_C_baseline = get_best_C(all_training, training_labels, c_values)
        if percentage == list_of_percentages[0]:

            # best_C_baseline=np.loadtxt(CV_best_param_folder + 'AwA' + "_svm_" + class_id + "class_"+ "%ddata_best.txt"%k)
            print 'getting best c for baseline'

            clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
            clf.fit(all_training, training_labels)

            filename='{}_baseline_fold{}.txt'.format(dataset,k)
            with open(os.path.join(CV_best_param_folder,filename),'a') as best_param_file:
                best_param_file.write(str(best_C_baseline))


            baseline_predictions = clf.predict(all_testing)
            with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as baseline_file:
                baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')




        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING



        c_svm_plus=best_C_baseline
        c_star_values = [1., 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]#, 0.00000001]
        # c_star_values = [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.]
        print 'getting best c star'
        # c_star_svm_plus = 10**-12


        c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training, c_svm_plus, c_star_values)

        with open(os.path.join(cross_validation_folder,'best_Cstar_param{}.txt'.format(k)),'a') as best_params_doc:
            best_params_doc.write("\n"+str(c_star_svm_plus))

        print 'c star', c_star_svm_plus, '\n'
        duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)
        lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()
        accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))
        with open(os.path.join(cross_validation_folder,'lupi-{}-{}.csv'.format(k,percentage)),'a') as cv_lupi_file:
            cv_lupi_file.write(str(accuracy_lupi)+',')


        print 'svm+ accuracy',(accuracy_lupi)


def univariate_selection(feats, labels, metric):
    selector = SelectPercentile(metric, percentile=100)
    selector.fit(feats, labels)
    scores = selector.pvalues_
    return np.array(np.argsort(scores))
# #
# for percentage in [5,10,25,50,75]:
#     single_fold(k=3, percentage=percentage, dataset='awa1', kernel='linear', cmin=0, cmax=7, number_of_cs=8,rank_metric=chi2)

single_fold(k=1, percentage=5, dataset='mushroom', kernel='linear', cmin=0, cmax=7, number_of_cs=8,rank_metric=chi2)