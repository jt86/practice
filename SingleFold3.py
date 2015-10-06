import os
import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import get_best_Cstar, get_best_C, get_best_RFE_C
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold

def single_fold(k, topk, dataset,datasetnum, kernel, cmin,cmax,number_of_cs, percent_of_priv=50):
        stepsize=100
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        outer_directory = get_full_path('Desktop/Privileged_Data/NIPS')
        # Check if output directory exists and make it if necessary
        output_directory = os.path.join(get_full_path(outer_directory),'not-fixedC-25-75-{}-{}-RFE-baseline-step={}-percent_of_priv={}bottom'.format(dataset,datasetnum,stepsize,percent_of_priv))
        print (output_directory)
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")

        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        try:
            os.makedirs(cross_validation_folder)
        except OSError:
            if not os.path.isdir(cross_validation_folder):
                raise


        all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset,datasetnum)

        if 'tech' in dataset:
            n_top_feats= topk
        else:
            n_top_feats = topk*all_training.shape[1]//100
        print ('n top feats',n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))
        ############

        CV_best_param_folder = os.path.join(output_directory,'{}CV/'.format(dataset))
        try:
            os.makedirs(CV_best_param_folder)
        except OSError:
            if not os.path.isdir(CV_best_param_folder):
                raise


        ########## GET BEST C FOR RFE

        print('getting best param for RFE')

        # best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize=stepsize)
        best_rfe_param=1

        # with open(os.path.join(cross_validation_folder,'best_rfe_param{}.txt'.format(k)),'a') as best_params_doc:
        #     best_params_doc.write("\n"+str(best_rfe_param))
        print('best rfe param', best_rfe_param)

        ###########  CARRY OUT RFE, GET ACCURACY
        # print ('test labels',testing_labels)
        svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=stepsize)
        print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        print (all_testing.shape,testing_labels.shape)
        print ('num of chosen feats',sum(x == 1 for x in rfe.support_))

        ACC = rfe.score(all_testing, testing_labels)
        best_n_mask = rfe.support_

        print ('\nrfe predictions',sum(x > 0 for x in rfe.predict(all_testing)),'of',len(all_testing))
        print ('rfe accuracy',ACC)

        with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,topk)),'a') as cv_svm_file:
            cv_svm_file.write(str(ACC)+",")

        ##############################  BASELINE - all features

        # best_C_baseline = get_best_C(all_training, training_labels, c_values)
        best_C_baseline=1


        if topk == 100 or topk == 5:
            clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
            clf.fit(all_training, training_labels)
            baseline_predictions = clf.predict(all_testing)
            print ('\nbaseline predictions',sum(x > 0 for x in baseline_predictions),'of',len(all_testing))
            # print ('baseline predictions',baseline_predictions)
            print ('baseline',accuracy_score(testing_labels,baseline_predictions))

            # filename='{}_baseline_fold{}.txt'.format(dataset,k)
            # with open(os.path.join(CV_best_param_folder,filename),'a') as best_param_file:
            #     best_param_file.write(str(best_C_baseline))
            #q
            with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as baseline_file:
                baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')



        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING
        print('doing svm+')
        normal_features_training = all_training[:,best_n_mask].copy()
        normal_features_testing = all_testing[:,best_n_mask].copy()
        privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()
        print('privileged',privileged_features_training.shape)


        all_features_ranking=rfe.ranking_
        print (all_features_ranking.shape)
        all_features_ranking = all_features_ranking[np.invert(best_n_mask)]
        print('\n\n\n\n all features ranking')
        print (all_features_ranking.shape)
        privileged_features_training = privileged_features_training[:,np.argsort(all_features_ranking)]

        num_of_priv_feats=percent_of_priv*privileged_features_training.shape[1]//100
        print('number to take', num_of_priv_feats)

        privileged_features_training = privileged_features_training[:,num_of_priv_feats:]
        print (privileged_features_training.shape)

        c_svm_plus=best_C_baseline
        c_star_values = [1., 0.1, 0.01, 0.001, 0.0001]#, 0.00001, 0.000001, 0.0000001, 0.00000001]
        c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training, c_svm_plus, c_star_values)
        with open(os.path.join(cross_validation_folder,'best_Cstar_param{}.txt'.format(k)),'a') as best_params_doc:
            best_params_doc.write("\n"+str(c_star_svm_plus))

        print('c star', c_star_svm_plus)
        duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)
        lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()
        # print ('lupi predictions',lupi_predictions)
        print ('\n lupi count',sum(x > 0 for x in lupi_predictions),'of',len(all_testing))
        accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))
        with open(os.path.join(cross_validation_folder,'lupi-{}-{}.csv'.format(k,topk)),'a') as cv_lupi_file:
            cv_lupi_file.write(str(accuracy_lupi)+',')


        print('svm+ accuracy',(accuracy_lupi))
#
#
# list_of_values = [5]#, 10, 25, 50, 75]
# # # # list_of_values = [300]#,400,500,600,700,800,900,1000]
# for dataset in ['arcene','madelon','dexter','dorothea']:
#     for top_k in list_of_values:
#         for i in range(10):#,11):
#             print ('\n\n NEW FOLD NUM {}'.format(i))
#             single_fold(k=i, topk=top_k, dataset=dataset, datasetnum=0, kernel='linear', cmin=0, cmax=4, number_of_cs=5, percent_of_priv=100)


