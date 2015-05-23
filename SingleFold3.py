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

def single_fold(k, dataset, kernel, cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):

        np.random.seed(k)

        c_values, cstar_values = get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin, cstarmax)
        print 'cvalues',c_values

        outer_directory = get_full_path('Desktop/Privileged_Data/')
        output_directory = os.path.join(get_full_path(outer_directory),'{}CV3'.format(dataset))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # original_features_array, labels_array, tuple = get_feats_and_labels(dataset)
        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
        # chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")




        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        if not os.path.exists(cross_validation_folder):
            os.makedirs(cross_validation_folder)#,exist_ok=True)


        if 'awa' in dataset:
            class_id =dataset[-1]
            all_training, all_testing, training_labels, testing_labels = get_awa_data("", class_id)
        else:
            all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset)



        total_number_of_feats = all_training.shape[1]
        list_of_percentages = [5,10,25,50,75]

        for percentage in list_of_percentages:
            topK = percentage/100
            n_top_feats=int(topK*total_number_of_feats)

            # n_top_feats = int(total_number_of_feats*percentage/100)

            param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))
            ############


            # method = 'privfeat_rfe_top'
            CV_best_param_folder = os.path.join(output_directory,'{}CV/'.format(dataset))
            if not os.path.exists(CV_best_param_folder):
                os.makedirs(CV_best_param_folder)



            print 'getting best param for RFE'

            # topK=percentage/100
            # best_rfe_param=np.loadtxt(CV_best_param_folder + 'AwA' + "_" + method + "_SVMRFE_%.2ftop"%topK+ "_" +class_id + "class_"+ "%ddata_best.txt"%k)
            # best_rfe_param=10.

            best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats)
            filename='{}_RFE_top{}_fold{}.txt'.format(dataset,percentage,k)
            with open(os.path.join(CV_best_param_folder,filename),'a') as best_param_file:
                best_param_file.write(str(best_rfe_param))
            print 'best rfe param', best_rfe_param

            ###########

            svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=1)
            rfe.fit(all_training, training_labels)
            ACC = rfe.score(all_testing, testing_labels)
            best_n_mask = rfe.support_

            with open(os.path.join(cross_validation_folder,'best_feats{}.txt'.format(k)),'a') as best_feats_doc:
                best_feats_doc.write("\n"+str(best_n_mask))


            with open(os.path.join(cross_validation_folder,'svm-{}.csv'.format(k)),'a') as cv_svm_file:
                cv_svm_file.write(str(ACC)+",")

            # ##############################  BASELINE - all features
            if percentage == list_of_percentages[0]:

                # best_C_baseline=np.loadtxt(CV_best_param_folder + 'AwA' + "_svm_" + class_id + "class_"+ "%ddata_best.txt"%k)
                # best_C_baseline=10.
                print 'getting best c for baseline'
                best_C_baseline = get_best_C(all_training, training_labels, c_values)
                clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
                clf.fit(all_training, training_labels)

                filename='{}_baseline_fold{}.txt'.format(dataset,k)
                with open(os.path.join(CV_best_param_folder,filename),'a') as best_param_file:
                    best_param_file.write(str(best_C_baseline))


                baseline_predictions = clf.predict(all_testing)
                with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as baseline_file:
                    baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')




            ############# SVM PLUS - PARAM ESTIMATION AND RUNNING

            normal_features_training = all_training[:,best_n_mask].copy()
            normal_features_testing = all_testing[:,best_n_mask].copy()
            privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()

            c_svm_plus=best_C_baseline
            c_star_values = [1., 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
            print 'getting best c star'
            c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training, c_svm_plus, c_star_values)
            print 'c star', c_star_svm_plus, '\n'
            duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)
            lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()
            accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))
            with open(os.path.join(cross_validation_folder,'lupi-{}.csv'.format(k)),'a') as cv_lupi_file:
                cv_lupi_file.write(str(accuracy_lupi)+',')


            print 'svm+ accuracy',(accuracy_lupi)


def get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):
    c_values = np.logspace(cmin,cmax,number_of_cs)
    if cstarmin==None:
        cstarmin, cstarmax = cmin,cmax
    cstar_values=np.logspace(cstarmin,cstarmax,number_of_cs)
    # c_values=np.array(c_values,dtype=int)
    return c_values, cstar_values

# for k in range (1,2):
#     single_fold(k=k, dataset='madelon', kernel='linear', cmin=-3, cmax=-1, number_of_cs=3)

# single_fold(k=1,dataset='madelon',kernel='linear',cmin=0,cmax=7, number_of_cs= 8)