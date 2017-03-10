'''
This is the main function.
Things to check before running: (1) values of C, (2) output directory and whether old output is there
(3) number of jobs in go-practice-submit.sh matches desired number of settings to run in Run Experiment
(4) that there is no code test run
(5) data is regularised as desired in GetSingleFoldData
(6) params including number of folds and stepsize set correctly
'''

import os
import numpy as np
from SVMplus import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from Get_Full_Path import get_full_path
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
import sys
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, mutual_info_classif, VarianceThreshold

def get_anova_ordered_indices(all_training,training_labels):

        # get array of scores in the same order as original features
        selector = SelectPercentile(f_classif, percentile=100)
        selector.fit(all_training, training_labels)
        scores = selector.scores_

        #sort the scores (small to big)
        sorted_indices_small_to_big = np.argsort(scores)
        sorted_scores = scores[sorted_indices_small_to_big]

        # nan indices are at the end. Reverse the score array (big to small) but leave these at the end
        nan_indices = np.argwhere(np.isnan(scores)).flatten()
        non_nan_indices = sorted_indices_small_to_big[:-len(nan_indices)]
        sorted_indices_big_to_small = non_nan_indices[::-1]
        ordered_indices = np.array(np.hstack([sorted_indices_big_to_small,nan_indices]))             # indices of biggest

        print(scores[ordered_indices])
        print ('ordered feats', ordered_indices.shape)
        return ordered_indices

def get_chi2_ordered_indices(all_training, training_labels):

        # add values so that all features of training data are non-zero
        min_value = np.min(all_training)
        all_training=all_training-min_value

        # get array of scores in the same order as original features
        selector = SelectPercentile(chi2, percentile=100)
        selector.fit(all_training, training_labels)
        scores = selector.scores_

        # sort the scores (small to big)
        ordered_indices = np.argsort(scores)[::-1]
        print(scores[ordered_indices])
        print(ordered_indices)
        return ordered_indices

def get_mutinfo_ordered_indices(all_training,training_labels):
        scores = mutual_info_classif(all_training, training_labels)
        ordered_indices = np.array(np.argsort(scores)[::-1])  # ordered feats is np array of indices from biggest to smallest
        return ordered_indices


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
        output_directory = get_full_path(('Desktop/Privileged_Data/{}-10x10-{}-ALLCV{}to{}-featsscaled-step{}-{}percentinstances/{}{}/top{}chosen-{}percentinstances/').format(featsel, dataset, cmin, cmax, stepsize, percentageofinstances, dataset, datasetnum, topk, percentageofinstances))
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


        if featsel == 'anova':
                ordered_indices = get_anova_ordered_indices(all_training,training_labels)
        if featsel == 'mutinfo':
                ordered_indices = get_mutinfo_ordered_indices(all_training, training_labels)
        if featsel == 'chi2':
                ordered_indices = get_chi2_ordered_indices(all_training, training_labels)
        # sys.exit()


        ######### GET TRAIN/TEST FOR NORMAL/PRIV

        sorted_training = all_training[:, ordered_indices]
        sorted_testing = all_testing[:, ordered_indices]

        normal_features_training = sorted_training[:, :n_top_feats]  # take the first n_top_feats
        normal_features_testing = sorted_testing[:, :n_top_feats]
        privileged_features_training = sorted_training[:, n_top_feats:]

        ######### BASELINE SVM WITH FEAT SELECTION

        best_param = get_best_C(all_training, training_labels, c_values, cross_validation_folder, datasetnum, topk)
        svc = SVC(C=best_param, kernel=kernel, random_state=k)
        svc.fit(normal_features_training, training_labels)
        svm_accuracy = svc.score(normal_features_testing, testing_labels)
        print('SVM accuracy (using slice):', svm_accuracy)

        with open(os.path.join(cross_validation_folder,'svm-{}-{}-{}.csv'.format(featsel,k,topk)),'a') as cv_svm_file:
            cv_svm_file.write(str(svm_accuracy)+",")

        # n_top_feats = topk*all_training.shape[1]//100
        print('n top feats', n_top_feats)
        param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats, k))


        for percent_of_priv in [100]:#,10]:
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


                c_star_values=c_values
                c_svm_plus,c_star_svm_plus = get_best_CandCstar(normal_features_training,training_labels, privileged_features_training2,
                                                 c_values, c_star_values,cross_validation_folder2,datasetnum, topk)

                duals,bias = svmplusQP(normal_features_training, training_labels.copy(), privileged_features_training2,  c_svm_plus, c_star_svm_plus)
                lupi_predictions = svmplusQP_Predict(normal_features_training,normal_features_testing ,duals,bias).flatten()

                accuracy_lupi = np.sum(testing_labels==np.sign(lupi_predictions))/(1.*len(testing_labels))

                with open(os.path.join(cross_validation_folder2,'lupi-{}-{}.csv'.format(k,topk)),'a') as cv_lupi_file:
                    cv_lupi_file.write(str(accuracy_lupi)+',')

                print ('k=',k, 'seed=',skfseed,'topk',topk,'svm+ accuracy=\n',accuracy_lupi)#'baseline accuracy=\n',accuracy_score(testing_labels,baseline_predictions))

dataset = 'tech'
for seed in range(1):
    for fold_num in range(10):  # 0
        for featsel in ['anova', 'chi2']:  # ,'mutinfo']:
            for top_k in [300]:  # ,500]:#,500]:#:,500]:#,500]:#,500]:#100,200,400,600,700,800,900,1000]:
                for take_top_t in ['top']:  # ,'bottom']:
                    for datasetnum in range(15,295,16):  # 5
                        # print (datasetnum)
                        single_fold(fold_num, 300, dataset, datasetnum, 'linear', -3, 3, 7, seed,100, 100, 'top', featsel)


# single_fold(k=3, topk=500, dataset='tech', datasetnum=245, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=4, percent_of_priv=100, percentageofinstances=100, take_top_t='bottom', featsel='chi2')

#
# def get_random_array(num_instances,num_feats):
#     random_array = np.random.rand(num_instances,num_feats)
#     random_array = preprocessing.scale(random_array)
#     return random_array

# value = 1

# for skfseed in range(10):
#     for k in range(10):
#         for datasetnum in range(211,242):
#             if datasetnum not in [254,174, 219, 197, 137]:
#                 single_fold(k=k, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,
#                             skfseed=skfseed, percent_of_priv=100, percentageofinstances=100, take_top_t='bottom', featsel='mutinfo')

# single_fold(k=3, topk=500, dataset='tech', datasetnum=245, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=4, percent_of_priv=100, percentageofinstances=100, take_top_t='bottom', featsel='chi2')

# for dataset in ['madelon','gisette','dexter','dorothea']:
#         for skfseed in range(10):
#             for k in range(10):
#                     single_fold(k=k, topk=300, dataset=dataset, datasetnum=None, kernel='linear', cmin=-3, cmax=3, number_of_cs=7,skfseed=skfseed, percent_of_priv=100, percentageofinstances=100,take_top_t='top')
#



# print(single_fold(k=0, topk=5000, dataset='awa', datasetnum=0, kernel='linear', cmin=-3, cmax=3, number_of_cs=4,skfseed=9, percent_of_priv=100, percentageofinstances=100,take_top_t='top'))