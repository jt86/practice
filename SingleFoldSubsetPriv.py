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

def single_fold(k, dataset, kernel, cmin,cmax,number_of_cs,amount_of_priv):
        step=1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        print 'cvalues',c_values

        outer_directory = get_full_path('Desktop/Privileged_Data/')


        output_directory = os.path.join(outer_directory,'{}-RFE-smallrange-baseline-subsetpriv{}'.format(dataset,4))
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)


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

        total_number_of_feats = all_training.shape[1]

        percentage=50
        topK = percentage/100
        n_top_feats=int(topK*total_number_of_feats)

        ############


        CV_best_param_folder = os.path.join(output_directory,'{}CV/'.format(dataset))
        try:
            os.makedirs(CV_best_param_folder)
        except OSError:
            if not os.path.isdir(CV_best_param_folder):
                raise

        print 'getting best param for RFE'
        best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats)
        with open(os.path.join(cross_validation_folder,'best_rfe_param{}.txt'.format(k)),'a') as best_params_doc:
            best_params_doc.write("\n"+str(best_rfe_param))

        ###########
        svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=step)
        rfe.fit(all_training, training_labels)
        best_n_mask = rfe.support_
        all_features_ranking=rfe.ranking_


        with open(os.path.join(cross_validation_folder,'best_feats{}.txt'.format(k)),'a') as best_feats_doc:
            best_feats_doc.write("\n"+str(best_n_mask))

        if amount_of_priv==10:
            ACC = rfe.score(all_testing, testing_labels)
            with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,percentage)),'a') as cv_svm_file:
                cv_svm_file.write(str(ACC)+",")

        # ##############################  BASELINE - all features
        print 'getting best c for baseline'
        best_C_baseline = get_best_C(all_training, training_labels, c_values)
        print 'got best C baseline',best_C_baseline


        if amount_of_priv==10:
            clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
            clf.fit(all_training, training_labels)
            filename='{}_baseline_fold{}.txt'.format(dataset,k)
            with open(os.path.join(CV_best_param_folder,filename),'a') as best_param_file:
                best_param_file.write(str(best_C_baseline))

            baseline_predictions = clf.predict(all_testing)
            with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as baseline_file:
                baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')




        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING
        # print 'ranking of all feats',all_features_ranking


        normal_features_testing = all_testing[:,best_n_mask].copy()
        normal_features_training = all_training[:,best_n_mask].copy()
        privileged_features_training = all_training[:,np.invert(best_n_mask)].copy()

        print amount_of_priv,'%'
        c_svm_plus=best_C_baseline
        c_star_values = [1., 0.1, 0.01, 0.001, 0.0001]

        print 'number of total priv feats',privileged_features_training.shape[1]
        number_of_priv_to_take= int(amount_of_priv*privileged_features_training.shape[1]/100)
        print 'number to take', number_of_priv_to_take


        all_features_ranking = all_features_ranking[np.invert(best_n_mask)]
        privileged_features_training = privileged_features_training[:,np.argsort(all_features_ranking)]

        privileged_features_training = privileged_features_training[:,:number_of_priv_to_take]
        print 'privileged',privileged_features_training.shape
        print 'sorted priv feature training', privileged_features_training.shape
        # print 'sorted priv feature training', privileged_features_training[0]

        c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training, c_svm_plus, c_star_values)
        with open(os.path.join(cross_validation_folder,'best_Cstar_param{}-subset{}.txt'.format(k,amount_of_priv)),'a') as best_params_doc:
            best_params_doc.write("\n"+str(c_star_svm_plus))
        print 'c star', c_star_svm_plus

        duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)

        lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()
        accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))
        with open(os.path.join(cross_validation_folder,'lupi-{}-subset{}.csv'.format(k,amount_of_priv)),'a') as cv_lupi_file:
            cv_lupi_file.write(str(accuracy_lupi)+',')


        print 'svm+ accuracy',(accuracy_lupi),'\n'
#
# subset_of_priv=40
# single_fold(7, 'mushroom', 'linear', 0,2,1,  subset_of_priv)
#
# for amount_of_priv in (10):#,20,30,40,50,60,70,80,90,100):
#     single_fold(1, 'heart', 'linear', 0,4,5,  amount_of_priv)

# '--k {} --subsetofpriv {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'