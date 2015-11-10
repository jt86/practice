import os
import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import get_best_Cstar, get_best_C, get_best_RFE_C, get_best_CandCstar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold


def single_fold(k, topk, dataset,datasetnum, kernel, cmin,cmax,number_of_cs, skfseed, percent_of_priv=100):


        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        print('cvalues',c_values)
        outer_directory = get_full_path('Desktop/Privileged_Data/10x10-CsCV-Cplus0.01-Cstarfinegrain-ALLCV/')#.format(c_star_svm_plus))
        output_directory = os.path.join(get_full_path(outer_directory),'fixedCandCstar-10fold-{}-{}-RFE-baseline-step={}-percent_of_priv={}'.format(dataset,datasetnum,stepsize,percent_of_priv))
        print (output_directory)
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



        n_top_feats = topk
        # if 'tech' in dataset:
        #     n_top_feats= topk
        # else:
        #     n_top_feats = topk*all_training.shape[1]//100
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

        best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize,cross_validation_folder,datasetnum,topk)
        # best_rfe_param=1

        # with open(os.path.join(cross_validation_folder,'best_rfe_param{}.txt'.format(k)),'a') as best_params_doc:
        #     best_params_doc.write("\n"+str(best_rfe_param))
        print('best rfe param', best_rfe_param)

        ###########  CARRY OUT RFE, GET ACCURACY

        svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=stepsize)
        print ('rfe step size',rfe.step)
        rfe.fit(all_training, training_labels)
        print (all_testing.shape,testing_labels.shape)
        print ('num of chosen feats',sum(x == 1 for x in rfe.support_))



        best_n_mask = rfe.support_
        normal_features_training = all_training[:,best_n_mask].copy()
        normal_features_testing = all_testing[:,best_n_mask].copy()
        privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()

        np.save('normal_features_training',normal_features_training)
        np.save('privileged_features_training',privileged_features_training)
        np.save('normal_features_testing',normal_features_testing)
        np.save('training_labels',training_labels)
        np.save('testing_labels',testing_labels)

        # print('all testing', all_testing.shape)
        print('testing labels', testing_labels.shape)


        svc = SVC(C=best_rfe_param, kernel="linear", random_state=1)
        svc.fit(normal_features_training,training_labels)
        rfe_accuracy = svc.score(normal_features_testing,testing_labels)
        print ('rfe accuracy (using slice):',rfe_accuracy)


        with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,topk)),'a') as cv_svm_file:
            cv_svm_file.write(str(rfe_accuracy)+",")
        ##############################  BASELINE - all features

        best_C_baseline = get_best_C(all_training, training_labels, c_values, cross_validation_folder,datasetnum,topk)
        # best_C_baseline=best_rfe_param
        print('all feats best c',best_C_baseline)

        print ('all training shape',all_training.shape)
        # if topk == 300 or topk == 5 or topk==10:
        clf = svm.SVC(C=best_C_baseline, kernel=kernel,random_state=1)
        clf.fit(all_training, training_labels)
        baseline_predictions = clf.predict(all_testing)
        print ('baseline',accuracy_score(testing_labels,baseline_predictions))

        with open(os.path.join(cross_validation_folder,'baseline-{}.csv'.format(k)),'a') as baseline_file:
            baseline_file.write (str(accuracy_score(testing_labels,baseline_predictions))+',')

        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING

        print('privileged',privileged_features_training.shape)
        all_features_ranking=rfe.ranking_
        print (all_features_ranking.shape)
        all_features_ranking = all_features_ranking[np.invert(best_n_mask)]
        privileged_features_training = privileged_features_training[:,np.argsort(all_features_ranking)]
        num_of_priv_feats=percent_of_priv*privileged_features_training.shape[1]//100
        print('number to take', num_of_priv_feats)
        privileged_features_training = privileged_features_training[:,:num_of_priv_feats]

        print ('privileged data shape',privileged_features_training.shape)

        # c_svm_plus=0.01
        # c_svm_plus=10
        c_star_values = [10., 5., 2., 1., 0.5, 0.2, 0.1]
        # c_star_values=[0.0001, 0.001, 0.01, 0.1]
        # c_star_values = np.logspace(-4,4,9)
        # print('c star values',c_star_values)
        # c_star_values=c_values
        # c_star_svm_plus=get_best_Cstar(normal_features_training,training_labels, privileged_features_training,
        #                                 c_svm_plus, c_star_values,cross_validation_folder,datasetnum, topk)
        #c_star_svm_plus=1.


        c_svm_plus, c_star_svm_plus = get_best_CandCstar(normal_features_training,training_labels, privileged_features_training,
                                         c_values, c_star_values,cross_validation_folder,datasetnum, topk)


        duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training,  c_svm_plus, c_star_svm_plus)
        lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()
        # print ('lupi predictions',lupi_predictions)
        # print ('\n lupi count',sum(x > 0 for x in lupi_predictions),'of',len(all_testing))
        accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))

        print('svm+ accuracy',(accuracy_lupi))
        with open(os.path.join(cross_validation_folder,'lupi-{}-{}.csv'.format(k,topk)),'a') as cv_lupi_file:
            cv_lupi_file.write(str(accuracy_lupi)+',')

        print ('c*',c_star_svm_plus)
        return (rfe_accuracy,accuracy_score(testing_labels,baseline_predictions),accuracy_lupi )

#
# num_folds=10
# mean_rfe,mean_all,mean_lupi = 0,0,0
# for fold in range(num_folds):
#     rfe_score, all_score,lupi_score =
#     mean_all+=all_score
#     mean_rfe+=rfe_score
#     mean_lupi+=lupi_score
# print (mean_all/num_folds,mean_rfe/num_folds,mean_lupi/num_folds)
#

# single_fold(k=1, topk=300, dataset='tech', datasetnum=48, kernel='linear', cmin=-2, cmax=2, number_of_cs=5,skfseed=1, percent_of_priv=100)