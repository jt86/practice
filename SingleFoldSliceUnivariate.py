'''
This is the main function.
Things to check before running: (1) values of C, (2) output directory and whether old output is there
(3) number of jobs in go-practice-submit.sh matches desired number of settings to run in Run Experiment
(4) that there is no code test run
(5) data is regularised as desired in GetSingleFoldData
(6) params including number of folds and stepsize set correctly
'''

import os

# print os.environ['HOME']

import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold
import sys
import numpy.random
from sklearn import preprocessing
# from time import time
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, mutual_info_classif, VarianceThreshold
# print (PYTHONPATH)




def single_fold(k, topk, dataset, datasetnum, kernel, cmin, cmax, number_of_cs, skfseed, percent_of_priv, percentageofinstances, take_top_t, featsel):

        if take_top_t not in ['top','bottom']:
                print('take top t should be "top"or "bottom"')
                sys.exit()

        print('using  {}% of training data instances'.format(percentageofinstances))
        # print('percentage of discarded info used as priv:{}'.format(percent_of_priv))
        stepsize=0.1
        np.random.seed(k)
        c_values = np.logspace(cmin,cmax,number_of_cs)
        print('cvalues',c_values)


        print('word',take_top_t)
        output_directory = get_full_path(('Desktop/Privileged_Data/LUFeSubset-{}-10x10-{}-ALLCV{}to{}-featsscaled-step{}-{}percentinstances/{}{}/top{}chosen-{}percentinstances/').format(featsel, dataset, cmin, cmax, stepsize, percentageofinstances, dataset, datasetnum, topk, percentageofinstances))
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

        # n_top_feats = topk*all_training.shape[1]//100
        print ('n top feats',n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))


        ######### UNIVARIATE PART

        ## ANOVA

        if featsel == 'anova':
                print('shapes:',all_training.shape,all_testing.shape)
                selector = VarianceThreshold()
                all_data = np.vstack([all_training,all_testing])
                # selector.fit(all_data)
                selector.fit_transform(all_training)
                non_neg_indices = selector.get_support()
                neg_indices = np.invert(non_neg_indices)


                print((non_neg_indices))
                print((neg_indices))
                print(len(non_neg_indices))
                print(len(neg_indices))

                sys.exit()

                all_training=all_training[:,non_neg_indices]
                all_testing = all_testing[:, non_neg_indices]


                print('shapes:', all_training.shape, all_testing.shape)

                selector = SelectPercentile(f_classif, percentile=100)
                selector.fit(all_training, training_labels)
                scores = selector.scores_
                print('scores', len(scores))
                print('scores', scores)



                print('sorted scores', scores[np.argsort(scores)[::-1]][:5000])
                ordered_feats = np.array(np.argsort(scores)[::-1])
                # ordered_feats = np.array(np.argsort(scores))
                print ('ordered feats', ordered_feats.shape)
                print('ordered feats', ordered_feats)

        ## MUTUAL INFO

        if featsel == 'mutinfo':
                scores = mutual_info_classif(all_training, training_labels)
                print('scores', len(scores))
                print('scores', scores)

                # [::-1]


                print('sorted scores',scores[np.argsort(scores)[::-1]]) # argsort(scores) gives them from smallest to biggest - indexing reverses this
                ordered_feats = np.array(np.argsort(scores)[::-1])    # ordered feats is np array of indices from biggest to smallest
                print('ordered feats', ordered_feats.shape)
                print('ordered feats', ordered_feats)


        sorted_training = all_training[:, ordered_feats]  # sort all instances' features from smallest to biggest
        sorted_testing = all_testing[:, ordered_feats]

        normal_features_training = sorted_training[:, :n_top_feats]  # take the first n_top_feats
        normal_features_testing = sorted_testing[:, :n_top_feats]
        privileged_features_training = sorted_training[:, n_top_feats:]

        # n_top_feats = topk*all_training.shape[1]//100
        print('n top feats', n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats, k))


        #
        # best_C = get_best_C(normal_features_training, training_labels, c_values, cross_validation_folder, datasetnum,
        #                     topk)
        # clf = svm.SVC(C=best_C, kernel=kernel, random_state=k)
        # clf.fit(normal_features_training, training_labels)
        # # svm_predictions = clf.predict(normal_features_testing)
        # # with open(os.path.join(cross_validation_folder, 'svm-{}-{}.csv'.format(k, topk)), 'a') as cv_svm_file:
        # #     cv_svm_file.write(str(accuracy_score(testing_labels, svm_predictions)) + ',')

        ########## GET BEST C FOR RFE
        #
        # # best_rfe_param = get_best_RFE_C(all_training,training_labels, c_values, n_top_feats,stepsize,cross_validation_folder,datasetnum,topk)
        # best_rfe_param = get_best_RFE_C(all_training, training_labels, c_values, n_top_feats, stepsize,
        #                                  datasetnum, topk)
        # print('best rfe param', best_rfe_param)

        ###########  CARRY OUT RFE, GET ACCURACY

        # svc = SVC(C=best_rfe_param, kernel=kernel, random_state=k)
        # rfe = RFE(estimator=svc, n_features_to_select=n_top_feats, step=stepsize)
        # print ('rfe step size',rfe.step)
        # rfe.fit(all_training, training_labels)
        # print (all_testing.shape,testing_labels.shape)
        # print ('num of chosen feats',sum(x == 1 for x in rfe.support_))
        #
        # best_n_mask = rfe.support_
        # normal_features_training = all_training[:,best_n_mask].copy()
        # normal_features_testing = all_testing[:,best_n_mask].copy()
        # privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()
        #
        # svc = SVC(C=best_rfe_param, kernel=kernel, random_state=k)
        # svc.fit(normal_features_training,training_labels)
        # rfe_accuracy = svc.score(normal_features_testing,testing_labels)
        # print ('rfe accuracy (using slice):',rfe_accuracy)
        #
        #
        # with open(os.path.join(cross_validation_folder,'svm-{}-{}.csv'.format(k,topk)),'a') as cv_svm_file:
        #     cv_svm_file.write(str(rfe_accuracy)+",")
        #
        # print('normal train shape {},priv train shape {}'.format(normal_features_training.shape,privileged_features_training.shape))
        # print('normal testing shape {}'.format(normal_features_testing.shape))



        ############# SVM PLUS - PARAM ESTIMATION AND RUNNING
        #
        # all_features_ranking = rfe.ranking_[np.invert(best_n_mask)]
        # privileged_features_training = privileged_features_training[:, np.argsort(all_features_ranking)]

        ##### THIS PART TO GET A SUBSET OF PRIV INFO####



        for percent_of_priv in [100,10]:
                num_of_priv_feats = percent_of_priv * privileged_features_training.shape[1] // 100
                print('privileged', privileged_features_training.shape)

                cross_validation_folder2 = os.path.join(cross_validation_folder,'{}-{}'.format(take_top_t, percent_of_priv))
                try:
                        os.makedirs(cross_validation_folder2)
                except OSError:
                        if not os.path.isdir(cross_validation_folder2):
                                raise


                if take_top_t=='top':
                        privileged_features_training2 = privileged_features_training[:,:num_of_priv_feats]
                if take_top_t=='bottom':
                        privileged_features_training2 = privileged_features_training[:,-num_of_priv_feats:]
                print ('privileged data shape',privileged_features_training2.shape)


        ##### THIS PART TO USE RANDOM DATA AS PRIVILEGED
        # privileged_features_training = get_random_array(privileged_features_training.shape[0],privileged_features_training.shape[1]*5)
        # random_array = np.random.rand(privileged_features_training.shape[0],privileged_features_training.shape[1])
        # random_array = preprocessing.scale(random_array)
        # privileged_features_training=random_array
        # print ('random data size',privileged_features_training.shape)
        #################################

                c_star_values=c_values
                c_svm_plus,c_star_svm_plus = get_best_CandCstar(normal_features_training,training_labels, privileged_features_training2,
                                                 c_values, c_star_values,cross_validation_folder2,datasetnum, topk)

                duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training2,  c_svm_plus, c_star_svm_plus)
                lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()

                accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))

                with open(os.path.join(cross_validation_folder2,'lupi-{}-{}.csv'.format(k,topk)),'a') as cv_lupi_file:
                    cv_lupi_file.write(str(accuracy_lupi)+',')

                print ('k=',k, 'seed=',skfseed,'topk',topk,'svm+ accuracy=\n',accuracy_lupi)#'baseline accuracy=\n',accuracy_score(testing_labels,baseline_predictions))



#
# def get_random_array(num_instances,num_feats):
#     random_array = np.random.rand(num_instances,num_feats)
#     random_array = preprocessing.scale(random_array)
#     return random_array

# value = 1

# single_fold(k=3, topk=500, dataset='tech', datasetnum=245, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=4, percent_of_priv=100, percentageofinstances=100, take_top_t='bottom', feat_sel='anova')

# for dataset in ['madelon','gisette','dexter','dorothea']:
#         for skfseed in range(10):
#             for k in range(10):
#                     single_fold(k=k, topk=300, dataset=dataset, datasetnum=None, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=skfseed, percent_of_priv=100, percentageofinstances=100,take_top_t='top')
#



# print(single_fold(k=0, topk=5000, dataset='awa', datasetnum=0, kernel='linear', cmin=-3, cmax=3, number_of_cs=4,skfseed=9, percent_of_priv=100, percentageofinstances=100,take_top_t='top'))