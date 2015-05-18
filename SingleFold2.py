from __future__ import division
import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import pairwise, accuracy_score
from sklearn import svm, linear_model
from SVMplus3 import svmplusQP, svmplusQP_Predict
from ParamEstimation2 import param_estimation
# from MainFunctionParallelised import get_indices_for_fold, get_train_test_selected_unselected, get_percentage_of_t, get_sorted_features
import sklearn.preprocessing
# from InitialFeatSelection import get_best_feats
import sklearn.preprocessing as preprocessing
from FeatSelection import get_ranked_indices, recursive_elimination2
from GetFeatsAndLabels import get_feats_and_labels
import argparse
from Get_Full_Path import get_full_path
from Get_Awa_Data import get_awa_data
from CollectBestParams import collect_best_rfe_param

from FromViktoriia import getdata


def single_fold(k, num_folds,dataset, peeking, kernel,
         cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):

        np.random.seed(k)
        rank_metric= 'r2'

        c_values, cstar_values = get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin, cstarmax)
        print 'cvalues',c_values

        output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/FixedCandCStar10/'),dataset)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        outer_directory = get_full_path('Desktop/Privileged_Data/FixedCandCStar11/')

        RFE_param_directory = os.path.join(get_full_path('Desktop/Privileged_Data/BestRFEParam/'),dataset)
        # RFE_param_directory = get_full_path('Desktop/Privileged_Data/BestRFEParam')

        # original_features_array, labels_array, tuple = get_feats_and_labels(dataset)
        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
        chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")

        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        if not os.path.exists(cross_validation_folder):
            os.makedirs(cross_validation_folder)#,exist_ok=True)


        list_of_t = []
        inner_folds = num_folds


        if k==0:
            with open(os.path.join(cross_validation_folder,'keyword.txt'),'a') as keyword_file:
                keyword_file.write("{} t values:{}\n peeking={}; {} folds; metric: {}; c={{10^{}..10^{}}}; c*={{10^{}..10^{}}} ({} values)".format(dataset,
                        list_of_t, peeking, num_folds, rank_metric, cmin, cmax, cstarmin, cstarmax, number_of_cs))

        if 'awa' in dataset:

            class_id =dataset[-1]
            all_training, all_testing, training_labels, testing_labels = get_awa_data("", class_id)






        else:
            print 'not awa'

def get_gamma_from_c(c_values, features):
    euclidean_distance = pairwise.euclidean_distances(features)
    median_euclidean_distance = np.median(euclidean_distance ** 2)
    return [value / median_euclidean_distance for value in c_values]


            ######################

        total_number_of_feats = all_training.shape[1]

                    #######################


        list_of_percentages = [5,10,25,50,75]
        for percentage in list_of_percentages:


            n_top_feats = int(total_number_of_feats*percentage/100)

            param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))

            #best_rfe_param = collect_best_rfe_param(k, percentage, RFE_param_directory)
            ############


            method = 'privfeat_rfe_top'
            PATH_CV_results = os.path.join(outer_directory,'CV/')
            topK=percentage/100

            print str((PATH_CV_results + 'AwA' + "_" + method + "_SVMRFE_%.2ftop"%topK+ "_" +class_id + "class_"+ "%ddata_best.txt"%k))
            best_rfe_param=np.loadtxt(PATH_CV_results + 'AwA' + "_" + method + "_SVMRFE_%.2ftop"%topK+ "_" +class_id + "class_"+ "%ddata_best.txt"%k)
            print 'best rfe param', best_rfe_param

            ###########


            best_n_mask = recursive_elimination2(all_training, training_labels, n_top_feats, best_rfe_param)
            with open(os.path.join(cross_validation_folder,'best_feats{}.txt'.format(k)),'a') as best_feats_doc:
                best_feats_doc.write("\n"+str(best_n_mask))
            normal_features_training = all_training[:,best_n_mask]
            normal_features_testing = all_testing[:,best_n_mask]
            privileged_features_training = all_training[:, np.invert(best_n_mask)]


            # ##############################  BASELINE - all features
            if percentage == list_of_percentages[0]:
                best_C_baseline=np.loadtxt(PATH_CV_results + 'AwA' + "_svm_" + class_id + "class_"+ "%ddata_best.txt"%k)
                param_estimation_file.write("\n\n Baseline scores array")

                # best_C_baseline = param_estimation(param_estimation_file, all_training,
                #                                                         training_labels, c_values,inner_folds, False, None,
                #                                                         peeking,testing_features=all_testing,
                #                                                         testing_labels=testing_labels)


                print 'best c baseline',best_C_baseline,  'kernel', kernel
                clf = svm.SVC(C=best_C_baseline, kernel=kernel)
                # pdb.set_trace()
                print all_training.shape, training_labels.shape
                clf.fit(all_training, training_labels)

                baseline_predictions = clf.predict(all_testing)
                with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as cv_baseline_file:
                    cv_baseline_file.write(str(accuracy_score(testing_labels, baseline_predictions))+",")



            ############################### SVM - PARAM ESTIMATION AND RUNNING

            param_estimation_file.write(
                # "\n\n SVM parameter selection for top " + str(n_top_feats) + " features\n" + "C,score")
                "\n\n SVM scores array for top " + str(n_top_feats) + " features\n")


            best_C_SVM  = param_estimation(param_estimation_file, normal_features_training,
                                          training_labels, c_values, inner_folds, privileged=False, privileged_training_data=None,
                                        peeking=peeking, testing_features=normal_features_testing,testing_labels=testing_labels)

            clf = svm.SVC(C=best_C_SVM, kernel=kernel)
            clf.fit(normal_features_training, training_labels)
            with open(os.path.join(cross_validation_folder,'svm-{}.csv'.format(k)),'a') as cv_svm_file:
                cv_svm_file.write(str(accuracy_score(testing_labels, clf.predict(normal_features_testing)))+",")


            ############# SVM PLUS - PARAM ESTIMATION AND RUNNING


            if n_top_feats != total_number_of_feats:
                assert n_top_feats < total_number_of_feats
                param_estimation_file.write(
                # "\n\n SVM PLUS parameter selection for top " + str(n_top_feats) + " features\n" + "C,C*,score")
                "\n\n SVM PLUS scores array for top " + str(n_top_feats) + " features\n")



                best_C_SVM_plus,  best_C_star_SVM_plus = 100, 1

                alphas, bias = svmplusQP(normal_features_training, training_labels.ravel(), privileged_features_training,
                                         best_C_SVM_plus, best_C_star_SVM_plus)


                LUPI_predictions_for_testing = svmplusQP_Predict(normal_features_training, normal_features_testing,
                                                                 alphas, bias, kernel).ravel()


                with open(os.path.join(cross_validation_folder,'lupi-{}.csv'.format(k)),'a') as cv_lupi_file:
                    cv_lupi_file.write(str(accuracy_score(testing_labels, LUPI_predictions_for_testing))+",")

            chosen_params_file.write("\n\n{} top features,fold {},baseline,{}".format(n_top_feats,k,best_C_baseline))
            chosen_params_file.write("\n,,SVM,{}".format(best_C_SVM))
            chosen_params_file.write("\n,,SVM+,{},{}" .format(best_C_SVM_plus,best_C_star_SVM_plus))






def get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):
    c_values = np.logspace(cmin,cmax,number_of_cs)
    if cstarmin==None:
        cstarmin, cstarmax = cmin,cmax
    cstar_values=np.logspace(cstarmin,cstarmax,number_of_cs)
    return c_values, cstar_values



# single_fold(k=1, num_folds=10, dataset='awa0', peeking=True, kernel='linear', cmin=0, cmax=5, number_of_cs=6)


