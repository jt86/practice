
import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import pairwise, accuracy_score
from sklearn import svm, linear_model
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import param_estimation,get_best_Cstar, get_best_C, get_best_RFE_C
from sklearn import svm, cross_validation
from Get_Full_Path import get_full_path
from Get_Awa_Data import get_awa_data
from CollectBestParams import collect_best_rfe_param
from sklearn.feature_selection import RFE
from FromViktoriia import getdata
import numpy
from sklearn.svm import SVC, LinearSVC
from GetSingleFoldData import get_train_and_test_this_fold

def single_fold(k, percentage, dataset,datasetnum, kernel, cmin,cmax,number_of_cs):

        np.random.seed(k)

        c_values = np.logspace(cmin,cmax,number_of_cs)
        print('cvalues',c_values)

        outer_directory = get_full_path('Desktop/Privileged_Data/')

        # Check if output directory exists and make it if necessary
        output_directory = os.path.join(get_full_path(outer_directory),'{}-{}-RFE-baseline-step=1000'.format(dataset,datasetnum))
        try:
            os.makedirs(output_directory)
        except OSError:
            if not os.path.isdir(output_directory):
                raise
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # original_features_array, labels_array, tuple = get_feats_and_labels(dataset)
        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
        # chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")




        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        try:
            os.makedirs(cross_validation_folder)
        except OSError:
            if not os.path.isdir(cross_validation_folder):
                raise

        #
        # # Deal with AwA dataset with OvO
        # if 'awa' in dataset:
        #     class_id =dataset[-1]
        #     all_training, all_testing, training_labels, testing_labels = get_awa_data("", class_id)
        # else:

        all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset,datasetnum)



        total_number_of_feats = all_training.shape[1]
        n_top_feats = int(total_number_of_feats*(percentage/100))

        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))
        ############


        # method = 'privfeat_rfe_top'
        CV_best_param_folder = os.path.join(output_directory,'{}CV/'.format(dataset))
        try:
            os.makedirs(CV_best_param_folder)
        except OSError:
            if not os.path.isdir(CV_best_param_folder):
                raise

        ##########  EXTRA TEST CLASSIFIER
        # svc = svm.SVC()
        # X_train, X_test = all_training, all_testing
        # y_train, y_test = training_labels, testing_labels
        # svc.fit(X_train,y_train)
        # print (X_test.shape, y_test.shape)
        # print ('standard svm score',svc.score(X_test, y_test))

        ########## RFE CROSS VALIDATION

        print('getting best param for RFE')
        stepsize=1000
        best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize=stepsize)
        # best_rfe_param=1

        # with open(os.path.join(cross_validation_folder,'best_rfe_param{}.txt'.format(k)),'a') as best_params_doc:
        #     best_params_doc.write("\n"+str(best_rfe_param))
        print('best rfe param', best_rfe_param)

        ###########  RFE CARRIED OUT; GET ACCURACY
        # print ('test labels',testing_labels)
        svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=stepsize)
        print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        ACC = rfe.score(all_testing, testing_labels)
        best_n_mask = rfe.support_
        print ('rfe accuracy',ACC)
        # with open(os.path.join(cross_validation_folder,'best_feats{}.txt'.format(k)),'a') as best_feats_doc:
        #     best_feats_doc.write("\n"+str(best_n_mask))
        #
        #
        with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,percentage)),'a') as cv_svm_file:
            cv_svm_file.write(str(ACC)+",")

        # ##############################  BASELINE - all features
        best_C_baseline = get_best_C(all_training, training_labels, c_values)
        list_of_percentages =[5, 10, 25, 50, 75]
        if percentage == list_of_percentages[0]:
        # if 1<2:

            # best_C_baseline=np.loadtxt(CV_best_param_folder + 'AwA' + "_svm_" + class_id + "class_"+ "%ddata_best.txt"%k)
            print('getting best c for baseline')

            clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
            clf.fit(all_training, training_labels)
            baseline_predictions = clf.predict(all_testing)
            # print ('baseline predictions',baseline_predictions)
            print ('baseline',accuracy_score(testing_labels,baseline_predictions))

            # filename='{}_baseline_fold{}.txt'.format(dataset,k)
            # with open(os.path.join(CV_best_param_folder,filename),'a') as best_param_file:
            #     best_param_file.write(str(best_C_baseline))
            #
            with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as baseline_file:
                baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')



        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING

        normal_features_training = all_training[:,best_n_mask].copy()
        normal_features_testing = all_testing[:,best_n_mask].copy()
        privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()


        c_svm_plus=best_C_baseline
        c_star_values = [1., 0.1, 0.01, 0.001, 0.0001]#, 0.00001, 0.000001, 0.0000001, 0.00000001]
        print('getting best c star')
        c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training, c_svm_plus, c_star_values)
        # c_star_svm_plus=0.1
        with open(os.path.join(cross_validation_folder,'best_Cstar_param{}.txt'.format(k)),'a') as best_params_doc:
            best_params_doc.write("\n"+str(c_star_svm_plus))

        print('c star', c_star_svm_plus, '\n')
        duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)
        lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()
        accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))
        with open(os.path.join(cross_validation_folder,'lupi-{}-{}.csv'.format(k,percentage)),'a') as cv_lupi_file:
            cv_lupi_file.write(str(accuracy_lupi)+',')


        print('svm+ accuracy',(accuracy_lupi))


# list_of_values = [5, 10, 25, 50, 75]

percentage = 50
for i in range(1):#,11):
    print ('\n\n NEW FOLD NUM {}'.format(i))
    single_fold(k=i, percentage=percentage, dataset='tech', datasetnum=0, kernel='linear', cmin=0, cmax=4, number_of_cs=5)
