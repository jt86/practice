from __future__ import division
import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import pairwise, accuracy_score
from sklearn import svm, linear_model
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import param_estimation, do_CV_svmrfe_5fold

from Get_Full_Path import get_full_path
from Get_Awa_Data import get_awa_data
from CollectBestParams import collect_best_rfe_param
from sklearn.feature_selection import RFE
from FromViktoriia import getdata
import numpy
from sklearn.svm import SVC, LinearSVC

def single_fold(k, num_folds,dataset, peeking, kernel,
         cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):

        np.random.seed(k)
        rank_metric= 'r2'

        c_values, cstar_values = get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin, cstarmax)
        # print 'cvalues',c_values

        outer_directory = get_full_path('Desktop/Privileged_Data/FixedCandCStar13/')
        output_directory = os.path.join(get_full_path(outer_directory),dataset)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)



        # RFE_param_directory = get_full_path('Desktop/Privileged_Data/BestRFEParam')

        # original_features_array, labels_array, tuple = get_feats_and_labels(dataset)
        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
        chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")




        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        if not os.path.exists(cross_validation_folder):
            os.makedirs(cross_validation_folder)#,exist_ok=True)


        list_of_t = []
        inner_folds = num_folds



        if 'awa' in dataset:

            class_id =dataset[-1]
            all_training, all_testing, training_labels, testing_labels = get_awa_data("", class_id)






        else:
            print 'not awa'

            ######################

        total_number_of_feats = all_training.shape[1]

                    #######################


        list_of_percentages = [5,10,25,50,75]
        # list_of_percentages = [75]
        for percentage in list_of_percentages:
            topK = percentage/100
            top=int(topK*total_number_of_feats)
            n_top_feats=top
            # n_top_feats = int(total_number_of_feats*percentage/100)

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


            estimator = svm.SVC(kernel="linear", C=best_rfe_param, random_state=1)
            selector = RFE(estimator, step=1, n_features_to_select=n_top_feats)
            selector = selector.fit(all_training, training_labels)
            best_n_mask = selector.support_

            svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=1)
            rfe.fit(all_training, training_labels)
            ACC = rfe.score(all_testing, testing_labels)
            selected = rfe.support_
            #
            # with open(os.path.join(output_directory, 'ACC_each_fold.csv'), "a")as scores_each_fold:
            #     scores_each_fold.write('\nfold num {}, {}%, {}'.format(k, percentage, ACC))

            best_n_mask = selected
            # numpy.set_printoptions(threshold=numpy.nan)
            # print 'mask'
            # print best_n_mask
            # sys.exit(0)
            with open(os.path.join(cross_validation_folder,'best_feats{}.txt'.format(k)),'a') as best_feats_doc:
                best_feats_doc.write("\n"+str(best_n_mask))
            normal_features_training = all_training[:,best_n_mask]
            normal_features_testing = all_testing[:,best_n_mask]
            privileged_features_training = all_training[:, np.invert(best_n_mask)]

            with open(os.path.join(cross_validation_folder,'svm-{}.csv'.format(k)),'a') as cv_svm_file:
                cv_svm_file.write(str(ACC)+",")
            # ##############################  BASELINE - all features
            if percentage == list_of_percentages[0]:
                best_C_baseline=np.loadtxt(PATH_CV_results + 'AwA' + "_svm_" + class_id + "class_"+ "%ddata_best.txt"%k)
                param_estimation_file.write("\n\n Baseline scores array")

                # best_C_baseline = param_estimation(param_estimation_file, all_training,
                #                                                         training_labels, c_values,inner_folds, False, None,
                #                                                         peeking,testing_features=all_testing,
                #                                                         testing_labels=testing_labels)


                print 'best c baseline',best_C_baseline,  'kernel', kernel
                clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
                # pdb.set_trace()
                print all_training.shape, training_labels.shape
                clf.fit(all_training, training_labels)

                baseline_predictions = clf.predict(all_testing)
                with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as baseline_file:
                    baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')


            ############################### SVM - PARAM ESTIMATION AND RUNNING
            #
            # param_estimation_file.write(
            #     # "\n\n SVM parameter selection for top " + str(n_top_feats) + " features\n" + "C,score")
            #     "\n\n SVM scores array for top " + str(n_top_feats) + " features\n")
            #
            # # best_C_SVM  = param_estimation(param_estimation_file, normal_features_training, training_labels, c_values, inner_folds)
            #
            # # (Xorig,Yorig, reg_array, top):
            # best_C_SVM = do_CV_svmrfe_5fold(normal_features_training,training_labels,c_values,n_top_feats)
            #
            #
            # # best_C_SVM  = param_estimation(param_estimation_file, normal_features_training,
            # #                               training_labels, c_values, inner_folds, privileged=False, privileged_training_data=None,
            # #                             peeking=peeking, testing_features=normal_features_testing,testing_labels=testing_labels)
            #
            # clf = svm.SVC(C=best_C_SVM, kernel=kernel,random_state=1)
            # clf.fit(normal_features_training, training_labels)
            # with open(os.path.join(cross_validation_folder,'svm-{}.csv'.format(k)),'a') as cv_svm_file:
            #     cv_svm_file.write(str(accuracy_score(testing_labels, clf.predict(normal_features_testing)))+",")


            ############# SVM PLUS - PARAM ESTIMATION AND RUNNING
                    #SVM+ part
            X_selected=all_training[:,selected].copy();
            test_X_selected=all_testing[:,selected].copy()
            notselected = numpy.invert(rfe.support_)
            X_priv=all_training[:,notselected].copy()
            #(reg_best, reg_best_star) = do_CV_svm_plus_5x5fold(X_selected, Y, X_priv, reg_array, reg_array, dataset, PATH_CV_results, method+"_%.2ftop"%topK, class_id, k)
            reg_best=100.
            reg_best_star=1.
            duals,bias = svmplusQP(X_selected,training_labels.copy(),X_priv,reg_best,reg_best_star)
            testXranked = svmplusQP_Predict(X_selected,test_X_selected,duals,bias).flatten()
            LUPI_ACC = numpy.sum(testing_labels==numpy.sign(testXranked))/(1.*len(testing_labels))
            with open(os.path.join(cross_validation_folder,'lupi-{}.csv'.format(k)),'a') as cv_lupi_file:
                cv_lupi_file.write(str(LUPI_ACC)+',')

            #
            # if n_top_feats != total_number_of_feats:
            #
            #     best_C_SVM_plus,  best_C_star_SVM_plus = 100, 1
            #
            #     alphas, bias = svmplusQP(normal_features_training, training_labels.ravel(), privileged_features_training,
            #                              best_C_SVM_plus, best_C_star_SVM_plus)
            #     # LUPI_predictions_for_testing = svmplusQP_Predict(normal_features_training, normal_features_testing,
            #     #                                                  alphas, bias, kernel).ravel()
            #
            #     ####
            #     testXranked = svmplusQP_Predict(normal_features_training,normal_features_testing,alphas,bias,kernel).flatten()
            #     LUPI_ACC = numpy.sum(testing_labels==numpy.sign(testXranked))/(1.*len(testing_labels))
            #     #
            #     # with open(os.path.join(output_directory, 'ACC_each_fold_LUPI.csv'), "a")as scores_each_fold:
            #     #     scores_each_fold.write('\nfold num {}, {}%, {}'.format(k, percentage, LUPI_ACC))
            #
            #     ###
            #

            #     # with open(os.path.join(output_directory, 'scores_each_fold_lupi.csv'), "a")as scores_each_fold_lupi:
            #     #     scores_each_fold_lupi.write('\nfold num {}, {}%, {}'.format(k, percentage, (accuracy_score(testing_labels, LUPI_predictions_for_testing))))

            #
            # chosen_params_file.write("\n\n{} top features,fold {},baseline,{}".format(n_top_feats,k,best_C_baseline))
            # # chosen_params_file.write("\n,,SVM,{}".format(best_C_SVM))
            # chosen_params_file.write("\n,,SVM+,{},{}" .format(best_C_SVM_plus,best_C_star_SVM_plus))


            print 'svm+ accuracy',(LUPI_ACC)







def get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):
    c_values = np.logspace(cmin,cmax,number_of_cs)
    if cstarmin==None:
        cstarmin, cstarmax = cmin,cmax
    cstar_values=np.logspace(cstarmin,cstarmax,number_of_cs)
    return c_values, cstar_values
#
# #
# for k in range (1,2):
#     single_fold(k=k, num_folds=10, dataset='awa3', peeking=False, kernel='linear', cmin=0, cmax=4, number_of_cs=5)
# # #
# # #
