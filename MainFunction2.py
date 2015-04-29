__author__ = 'jt306'

import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import f1_score, pairwise
from sklearn import svm, linear_model
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from SVMplus import svmplusQP, svmplusQP_Predict
import sklearn.preprocessing as preprocessing
from Get_Figures import get_figures
from ParamEstimation2 import param_estimation
from FeatSelection import univariate_selection, recursive_elimination
import time
from FeatSelection import get_ranked_indices, recursive_elimination2
from InitialFeatSelection import get_best_feats
from Get_Mean import get_mean_from, get_error_from
import pdb


def main_function(original_features_array, labels_array, output_directory, num_folds,
                  tuple, cmin, cmax,number_of_cs, peeking, dataset, rank_metric, prop_priv=1, multiplier=1, bottom_n_percent=0,
                logger=None, cstar_values=None,  cstarmin=None, cstarmax=None):

    results_file = open(os.path.join(output_directory, 'output.csv'), "a")
    param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
    chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")

    c_values = np.logspace(cmin,cmax,number_of_cs)
    if cstarmin==None:
        cstarmin, cstarmax = cmin,cmax
    cstar_values=np.logspace(cstarmin,cstarmax,number_of_cs)


    print 'c values', c_values, 'cstarvalues', cstar_values

    total_num_items = original_features_array.shape[0]
    original_number_feats = original_features_array.shape[1]

    # numbers_of_features_list = []
    list_of_t = []
    all_folds_SVM, all_folds_LUPI = [[]]*num_folds,[[]]*num_folds
    baseline_score, baseline_score2 =[], []

    k = -1
    for train,test in StratifiedKFold(labels_array, num_folds, shuffle=False , random_state=1):
        k+=1
        print'k',k

        total_number_of_items = (train.shape[0])
        number_of_training_instances = int(total_number_of_items - (total_number_of_items / num_folds)) - 1


        all_training, all_testing = original_features_array[train], original_features_array[test]
        training_labels, testing_labels = labels_array[train], labels_array[test]

        assert all_training.shape[0]+all_testing.shape[0] == original_features_array.shape[0], 'training + testing = total'

        rs = ShuffleSplit((number_of_training_instances - 1), n_iter=10, test_size=.2, random_state=0)
        # rs = StratifiedShuffleSplit(y=training_labels, n_iter=num_folds, test_size=.2, random_state=0)

        top_t_indices, remaining_indices = get_best_feats(all_training,training_labels,c_values, num_folds, rs)

        top_t_training, unselected_features_training = all_training[:,top_t_indices], all_training[:,remaining_indices]
        top_t_testing, unselected_features_testing = all_testing[:,top_t_indices], all_testing[:,remaining_indices]

        assert top_t_testing.shape[1]+unselected_features_testing.shape[1]==original_features_array.shape[1],'top t + remaining feats = total num of feats'
        # assert original_features_array.shape[1]/num_folds==top_t_testing.shape[1],'same size'



        print top_t_training.shape, unselected_features_training.shape, top_t_testing.shape, unselected_features_testing.shape

        t = top_t_training.shape[1]
        list_of_t.append(t)

        number_remaining_feats = original_number_feats - (bottom_n_percent * (original_number_feats) / 100)

        # numbers_of_features_list = []
        fold_results_SVM, fold_results_LUPI   =  [], []

        n = -1
        #
        # if t <= 15:
        #     tuple = [1,t+1,1]
        # else:
        #     tuple  = [t/10, t+1, t/10]
        #
        #
        # list_of_values = [i for i in range(*tuple)]


        list_of_values = get_percentage_of_t(t)[0]
        list_of_percentages = get_percentage_of_t(t)[1]

        for n_top_feats in list_of_values:
            #
            # if n_top_feats not in numbers_of_features_list:
            #     numbers_of_features_list.append(n_top_feats)

            n+=1
            param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))
             # Get T, Get NORMAL and PRIVILEGED

            top_t_sorted_training = get_sorted_features(top_t_training, training_labels, rank_metric, n_top_feats, dataset, logger, number_remaining_feats)
            top_t_sorted_testing = get_sorted_features(top_t_testing, testing_labels, rank_metric, n_top_feats, dataset, logger, number_remaining_feats)

            normal_indices = np.arange(0, n_top_feats)
            privileged_indices = [index for index in range(t) if index not in normal_indices]

            normal_features_training = top_t_sorted_training[:,normal_indices]
            normal_features_testing = top_t_sorted_testing[:,normal_indices]
            privileged_features_training = top_t_sorted_training[:, privileged_indices]
            privileged_features_training = np.hstack([privileged_features_training,unselected_features_training])
            # privileged_features_training = privileged_features_training[:(privileged_features_training.shape[1]/prop_priv)]

            print 'top t sorted shape', top_t_sorted_training.shape

            number_of_training_instances = int(total_number_of_items - (total_number_of_items / num_folds)) - 1
            # ##############################  BASELINE - all features



            if n_top_feats == list_of_values[0]:

                param_estimation_file.write("\n\n Baseline scores array")

                rs = StratifiedShuffleSplit(y=training_labels, n_iter=10, test_size=.2, random_state=0)
                best_C_baseline = param_estimation(param_estimation_file, all_training,
                                                                        training_labels, c_values,rs, False, None,
                                                                        peeking,testing_features=all_testing,
                                                                        testing_labels=testing_labels, logger=logger)[0]


                clf = svm.SVC(C=best_C_baseline, kernel='linear')#, gamma=best_gamma_baseline)
                # pdb.set_trace()
                clf.fit(all_training, training_labels)

                baseline_predictions = clf.predict(all_testing)
                baseline_score.append(f1_score(testing_labels, baseline_predictions))

                print 'baseline score length', len(baseline_score)
                print '\n\n baseline score', baseline_score


                # ##############################  BASELINE 2 - top t features only

                # rs = ShuffleSplit((number_of_training_instances - 1), n_iter=num_folds, test_size=.2, random_state=0)
                rs = StratifiedShuffleSplit(y=training_labels, n_iter=10, test_size=.2, random_state=0)
                best_C_baseline2 = param_estimation(param_estimation_file, top_t_training,
                                                                        training_labels, c_values,rs, False, None,
                                                                        peeking,testing_features=top_t_testing,
                                                                        testing_labels=testing_labels, logger=logger)[0]


                clf2 = svm.SVC(C=best_C_baseline2, kernel='linear')#, gamma=best_gamma_baseline)
                clf2.fit(top_t_training, training_labels)

                baseline_predictions2 = clf2.predict(top_t_testing)
                baseline_score2.append(f1_score(testing_labels, baseline_predictions2))

            ############################### SVM - PARAM ESTIMATION AND RUNNING
            total_number_of_items = len(train)
            # rs = ShuffleSplit((number_of_training_instances - 1), n_iter=num_folds, test_size=.2, random_state=0)
            rs = StratifiedShuffleSplit(y=training_labels, n_iter=10, test_size=.2, random_state=0)
            param_estimation_file.write(
                # "\n\n SVM parameter selection for top " + str(n_top_feats) + " features\n" + "C,score")
                "\n\n SVM scores array for top " + str(n_top_feats) + " features\n")


            best_C_SVM  = param_estimation(param_estimation_file, normal_features_training,
                                          training_labels, c_values, rs, privileged=False, privileged_training_data=None,
                                        peeking=peeking, testing_features=normal_features_testing,testing_labels=testing_labels, logger=logger)[0]

            clf = svm.SVC(C=best_C_SVM, kernel='linear')
            clf.fit(normal_features_training, training_labels)
            fold_results_SVM.append(f1_score(testing_labels, clf.predict(normal_features_testing)))



            ############# SVM PLUS - PARAM ESTIMATION AND RUNNING
            rs = ShuffleSplit((number_of_training_instances - 1), n_iter=10, test_size=.2, random_state=0)
            rs = StratifiedShuffleSplit(y=training_labels, n_iter=num_folds, test_size=.2, random_state=0)
            if n_top_feats != number_remaining_feats:
                param_estimation_file.write(
                # "\n\n SVM PLUS parameter selection for top " + str(n_top_feats) + " features\n" + "C,C*,score")
                "\n\n SVM PLUS scores array for top " + str(n_top_feats) + " features\n")
                best_C_SVM_plus,  best_C_star_SVM_plus  = param_estimation(
                param_estimation_file, normal_features_training, training_labels, c_values, rs, privileged=True,
                privileged_training_data=privileged_features_training,peeking=peeking,
                testing_features=normal_features_testing, testing_labels=testing_labels,
                multiplier=multiplier, logger=logger, cstar_values=cstar_values)

                alphas, bias = svmplusQP(normal_features_training, training_labels.ravel(), privileged_features_training,
                                         best_C_SVM_plus, best_C_star_SVM_plus)


                LUPI_predictions_for_testing = svmplusQP_Predict(normal_features_training, normal_features_testing,
                                                                 alphas, bias).ravel()

                fold_results_LUPI.append(f1_score(testing_labels, LUPI_predictions_for_testing))


            chosen_params_file.write("\n\n{} top features,fold {},baseline,{}".format(n_top_feats,k,best_C_baseline))
            chosen_params_file.write("\n,,SVM,{}".format(best_C_SVM))
            chosen_params_file.write("\n,,SVM+,{},{}" .format(best_C_SVM_plus,best_C_star_SVM_plus))


        all_folds_SVM[k] = fold_results_SVM
        all_folds_LUPI[k] = fold_results_LUPI


        print all_folds_SVM

    print 'all folds SVM length', len(all_folds_SVM)

    baseline_results = [baseline_score] * len(list_of_values)
    baseline_results2 = [baseline_score2] * len(list_of_values)

    keyword = "{} ({}x{}) t values:{}\n peeking={}; {} folds; metric: {}; c={{10^{}..10^{}}}; c*={{10^{}..10^{}}} ({} values)".format(dataset,
                   total_num_items, original_number_feats, list_of_t, peeking, num_folds, rank_metric, cmin, cmax, cstarmin, cstarmax, number_of_cs)

    get_figures(list_of_percentages, all_folds_SVM, all_folds_LUPI, baseline_results, baseline_results2, num_folds, output_directory, keyword)


def cut_off_arcene(sorted_features,logger):
    sorted_features = sorted_features[:, :9920]
    logger.info('Using arcene dataset: discarding bottom 80 features that consist of 0s')
    return sorted_features


def discard_bottom_n(sorted_features, number_remaining_feats, logger):
    logger.info("Shape before discard %r", sorted_features.shape)
    sorted_features = sorted_features[:, :number_remaining_feats]
    logger.info("Shape after discard %r", sorted_features.shape)
    return sorted_features

def take_top_feats(privileged_indices,prop_priv,logger):
    if len(privileged_indices) > 1:
        number_of_indices_to_take = len(privileged_indices) / prop_priv
        logger.info("number_of_indices_to_take: %d of %d", number_of_indices_to_take, len(privileged_indices))
        privileged_indices = privileged_indices[:number_of_indices_to_take]
    else:
        logger.info("1 or less indices remaining. Took %d", len(privileged_indices))
    return privileged_indices

def get_sorted_features(features_array, labels_array, rank_metric, n_top_feats,dataset, logger, number_remaining_feats):
    sorted_features = features_array[:,np.argsort(get_ranked_indices(features_array, labels_array, rank_metric, n_top_feats))]
    logger.info('sorted feats shape: %r', sorted_features.shape)


    if dataset == 'arcene' and sorted_features.shape[1] > 9000:
        sorted_features = cut_off_arcene(sorted_features, logger)

    number_of_samples = sorted_features.shape[0]
    # scaler = preprocessing.StandardScaler().fit(sorted_features)
    # sorted_features = scaler.transform(sorted_features, number_of_samples)
    # sorted_features = preprocessing.scale(sorted_features, axis=1)
    normaliser = preprocessing.Normalizer('l1')
    normaliser.fit_transform(sorted_features)
    sorted_features = discard_bottom_n(sorted_features, number_remaining_feats, logger)

    logger.info('first item in r2 sorted feats %r',sorted_features[0])
    return sorted_features


def get_percentage_of_t(t, tuple=(10,101,15)):
    list_of_values, list_of_percentages =[],[]
    for i in range(*tuple):
        print i
        list_of_values.append(t*i/100)
        list_of_percentages.append(i)
    return list_of_values,list_of_percentages