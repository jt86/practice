__author__ = 'jt306'

import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from sklearn.metrics import f1_score, pairwise
from sklearn import svm, linear_model
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from SVMplus import svmplusQP, svmplusQP_Predict
import sklearn.preprocessing as preprocessing
from Get_Figures import get_figures
from ParamEstimation import param_estimation
from FeatSelection import univariate_selection, recursive_elimination
import time
from FeatSelection import get_ranked_indices, recursive_elimination2
from InitialFeatSelection import get_best_feats
import pdb


def main_function(original_features_array, labels_array, output_directory, num_folds,
                  tuple, cmin, cmax, peeking, dataset, rank_metric,init_folds, prop_priv=1, multiplier=1, bottom_n_percent=0, logger=None, ):
    
    logger.debug('peeking status: %s', peeking)
    logger.info('main function: dataset = %s, peeking=%s ', dataset, peeking)

    c_values = np.logspace(cmin,cmax, num =7)
    print c_values

    top_t_features_array,unselected_features_array = get_best_feats(original_features_array,labels_array,c_values, init_folds)
    print top_t_features_array.shape, unselected_features_array.shape

    print "top t feat array shape", top_t_features_array.shape
    t = top_t_features_array.shape[1]
    original_number_feats = original_features_array.shape[1]

    print "t=",t
    if dataset == 'arcene' and original_number_feats >9999:
        original_number_feats = 9920

    if t <= 15:
        tuple = [1,t+1,1]
    else:
        tuple  = [t/10, t+1, t/10]

    print 'tuple',tuple
    # sys.exit(0)
    results_file = open(os.path.join(output_directory, 'output.csv'), "a")
    param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
    chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")

    number_remaining_feats = original_number_feats - (bottom_n_percent * (original_number_feats) / 100)
    logger.info('number of remaining features: %d', number_remaining_feats)


    # ####################
    # if rank_metric != 'r2':
    #
    #     sorted_features = features_array[:, get_ranked_indices(features_array, labels_array, rank_metric)]
    #     logger.debug('first item in features array before sorting \n %r', features_array[0])
    #
    #     logger.debug('first item in sorted features \n %r', sorted_features[0])
    #
    #     number_of_samples = sorted_features.shape[0]
    #     scaler = preprocessing.StandardScaler().fit(sorted_features)
    #     sorted_features = scaler.transform(sorted_features, number_of_samples)
    #     sorted_features = preprocessing.scale(sorted_features, axis=1)
    #
    #
    #     if dataset == 'arcene' and sorted_features.shape[1] > 9000:
    #         sorted_features = cut_off_arcene(sorted_features, logger)

        #
        #     # ########### CURRENT NEW BIT ######### - this is for non-r2 - need for r2 too
        #
        # sorted_features = discard_bottom_n(sorted_features, number_remaining_feats, logger)
        # logger.debug('first item after cutting off \n %r', sorted_features[0])

    numbers_of_features_list = []
    results, LUPI_results, baseline_results = [], [], []
    baseline_score, baseline_score2 = [], []


    logger.info('full list of values %r', range(*tuple))
    list_of_values = [i for i in range(*tuple)]
    logger.info('kept %d of %d values in list', len(list_of_values), len (range(*tuple))+1)
    logger.info("list of values %r", list_of_values)


    for n_top_feats in list_of_values:

        if rank_metric == 'r2':
            top_t_sorted = get_sorted_features(top_t_features_array, labels_array, rank_metric, n_top_feats, dataset, logger, number_remaining_feats)
            sorted_original = get_sorted_features(original_features_array, labels_array, rank_metric, n_top_feats,dataset, logger, number_remaining_feats)

        # ##### THIS BIT TO GET NORMAL AND PRIVILEGED FEATURES ##########

        normal_indices = np.arange(0, n_top_feats)
        logger.info('\n\n\n NORMAL INDICES \n %r', normal_indices)
        print 't', t
        print 'range t ', range(t)
        privileged_indices = [index for index in range(t) if index not in normal_indices]

        print "normal_indices", normal_indices
        print "privileged indices", privileged_indices

        # deals with prop_priv variable
        privileged_indices = take_top_feats(privileged_indices,prop_priv,logger)

        normal_features = top_t_sorted[:, normal_indices]
        privileged_features = top_t_sorted[:, privileged_indices]
        print 'privileged shape', privileged_features.shape

        # privileged_features = np.hstack([privileged_features,unselected_features_array])
        print 'privileged shape', privileged_features.shape

        SVM_score = []
        LUPI_score = []

        skf = StratifiedKFold(labels_array, num_folds)

        k = -1

        for train, test in skf:
            k += 1
            param_estimation_file.write("\n\n\n n=" + str(n_top_feats) + " fold = " + str(k))

            logger.info("\n \n Top %d features; Fold number %d", n_top_feats, k)

            normal_training_data, normal_testing_data = normal_features[train], normal_features[test]
            privileged_training_data, privileged_testing_data = privileged_features[train], privileged_features[test]
            all_training_data, all_testing_data = top_t_sorted[train], top_t_sorted[test]

            training_labels, testing_labels = labels_array[train], labels_array[test]

            number_of_items = (top_t_sorted.shape[0])
            number_of_training_instances = int(number_of_items - (number_of_items / num_folds)) - 1
            # logger.info( number_of_training_instances)

            rs = ShuffleSplit((number_of_training_instances - 1), n_iter=num_folds, test_size=.2, random_state=0)

            if n_top_feats == list_of_values[0]:

                logger.info("First iteration, doing param selection for baseline")

                # ##############################  BASELINE

                param_estimation_file.write("\n\n Baseline parameter selection \n C,gamma,score")
                best_C_baseline = param_estimation(param_estimation_file, all_training_data,
                                                                        training_labels, c_values,rs, False, None,
                                                                        peeking=peeking,testing_features=all_testing_data,
                                                                        testing_labels=testing_labels, logger=logger)[0]


                clf = svm.SVC(C=best_C_baseline, kernel='linear')#, gamma=best_gamma_baseline)
                # pdb.set_trace()
                clf.fit(all_training_data, training_labels)

                baseline_predictions = clf.predict(all_testing_data)
                baseline_score.append(f1_score(testing_labels, baseline_predictions))


                # ##############################  BASELINE 2 - all features


                best_C_baseline2 = param_estimation(param_estimation_file, all_training_data,
                                                                        training_labels, c_values,rs, False, None,
                                                                        peeking=peeking,testing_features=all_testing_data,
                                                                        testing_labels=testing_labels, logger=logger)[0]


                clf2 = svm.SVC(C=best_C_baseline2, kernel='linear')#, gamma=best_gamma_baseline)
                clf2.fit(all_training_data, training_labels)

                baseline_predictions2 = clf.predict(all_testing_data)
                baseline_score2.append(f1_score(testing_labels, baseline_predictions2))

            ############################### SVM - PARAM ESTIMATION AND RUNNING

            param_estimation_file.write(
                "\n\n SVM parameter selection for top " + str(n_top_feats) + " features\n" + "C,gamma,score")


            best_C_SVM  = param_estimation(param_estimation_file, normal_training_data,
                                                          training_labels, c_values,
                                                          rs, privileged=False, privileged_training_data=None,
                                                          peeking=peeking, testing_features=normal_testing_data,
                                                          testing_labels=testing_labels, logger=logger)[0]

            clf = svm.SVC(C=best_C_SVM, kernel='linear')
            clf.fit(normal_training_data, training_labels)
            SVM_score.append(f1_score(testing_labels, clf.predict(normal_testing_data)))


            # ############ SVM PLUS - PARAM ESTIMATION AND RUNNING


            if n_top_feats != number_remaining_feats:

                param_estimation_file.write(("\n\n SVM+ parameter selection{} top features:,C,C*").format(n_top_feats))

                best_C_SVM_plus,  best_C_star_SVM_plus  = param_estimation(
                    param_estimation_file, normal_training_data, training_labels, c_values, rs, privileged=True,
                    privileged_training_data=privileged_training_data,peeking=peeking,
                    testing_features=normal_testing_data, testing_labels=testing_labels,
                    multiplier=multiplier, logger=logger)


                logger.info('SVM+ feat selection finished. Doing SVM+')

                alphas, bias = svmplusQP(normal_training_data, training_labels.ravel(),
                                         privileged_training_data, best_C_SVM_plus, best_C_star_SVM_plus)


                LUPI_predictions_for_testing = svmplusQP_Predict(normal_training_data, normal_testing_data,
                                                                 alphas, bias).ravel()

                LUPI_score.append(f1_score(testing_labels, LUPI_predictions_for_testing))
            print "\n\n"

            logger.info("best baseline params: {}".format(best_C_baseline))
            logger.info("best SVM params: {}".format(best_C_SVM))
            logger.info("best SVM+ params: {}, {}".format(best_C_SVM_plus, best_C_star_SVM_plus))



            chosen_params_file.write("\n\n{} top features,fold {},baseline,{}".format(n_top_feats,k,best_C_baseline))
            chosen_params_file.write("\n,,SVM,{}".format(best_C_SVM))
            chosen_params_file.write("\n,,SVM+,{},{}" .format(best_C_SVM_plus,best_C_star_SVM_plus))




        results.append(SVM_score)  # put score and feature rank into an array
        if n_top_feats != number_remaining_feats:
            LUPI_results.append(LUPI_score)
        numbers_of_features_list.append(n_top_feats)



    # logger.info( "baseline_score ", baseline_score)
    baseline_results = [baseline_score] * len(numbers_of_features_list)
    baseline_results2 = [baseline_score] * len(numbers_of_features_list)


    keyword = "{} ({}x{}) \n peeking={}; {} folds; rank metric: {}; c range: 10^{} to 10^{}; init folds:{}".format(dataset,
                   original_features_array.shape[0], original_number_feats, peeking, num_folds, rank_metric, cmin, cmax, init_folds)
    # bottom {}% rejected, bottom_n_percent

    logger.info('LUPI LENGTH %d', len(LUPI_results))
    logger.info('normal length %d', len(results))
    logger.info('num feats list %d', len(numbers_of_features_list))
    get_figures(numbers_of_features_list, results, LUPI_results, baseline_results, baseline_results2, num_folds, output_directory, keyword)

    results_file.write(str(np.mean(baseline_results, axis=1)))
    results_file.write(str(np.mean(results, axis=1)))
    results_file.write(str(np.mean(LUPI_results, axis=1)))


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
    sorted_features = preprocessing.scale(sorted_features, axis=1)
    sorted_features = discard_bottom_n(sorted_features, number_remaining_feats, logger)

    logger.info('first item in r2 sorted feats %r',sorted_features[0])
    return sorted_features