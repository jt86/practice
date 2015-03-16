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




#
# sys.exit(0)


def main_function(features_array, labels_array,output_directory, num_folds,
                  tuple, c_values, peeking, dataset, rank_metric, prop_priv=1, multiplier=1, bottom_n_percent=0):


    print 'main function: dataset =', dataset,' peeking  = ',peeking


    number_of_features = features_array.shape[1]

    results_file = open(os.path.join(output_directory, 'output.csv'), "a")
    param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
    chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")

    total_num_feats = number_of_features
    if dataset == 'arcene':
        total_num_feats = 9920


    number_remaining_feats = number_of_features-(bottom_n_percent*(features_array.shape[1])/100)
    number_rejected_feats = number_of_features - number_remaining_feats
    print 'rejected',number_rejected_feats
    print 'number of remaining features', number_remaining_feats
    # sys.exit(0)

    # ####################
    if rank_metric != 'r2':

        sorted_features = features_array[:, get_ranked_indices(features_array,labels_array,rank_metric)]
        print 'first item in features array before sorting \n',features_array[0]

        print 'first item in sorted features \n',sorted_features[0]



        number_of_samples = sorted_features.shape[0]
        scaler = preprocessing.StandardScaler().fit(sorted_features)
        sorted_features = scaler.transform(sorted_features, number_of_samples)
        sorted_features = preprocessing.scale(sorted_features, axis=1)
        number_of_features = sorted_features.shape[1]



        if dataset == 'arcene' and sorted_features.shape[1] >9000:
            sorted_features,number_of_features = cut_off_arcene(sorted_features)


                                                                            ############ CURRENT NEW BIT ######### - this is for non-r2 - need for r2 too

        sorted_features,number_of_features = discard_bottom_n(sorted_features,number_remaining_feats)
        print 'first item after cutting off', sorted_features[0]





    numbers_of_features_list = []
    results, LUPI_results, baseline_results = [], [], []
    baseline_score = []


    full_list_of_values = range(*tuple)+[number_of_features]
    print 'full list of values',full_list_of_values
    print number_remaining_feats

    if number_remaining_feats==number_of_features:
        list_of_values = full_list_of_values
    else:
        # list_of_values = full_list_of_values[full_list_of_values<100]
        list_of_values = [i for i in full_list_of_values if i <=number_remaining_feats]
        print 'list of values',list_of_values
        # sys.exit(0)

    for n_top_feats in list_of_values:

        if rank_metric == 'r2':

            sorted_features = features_array[:,np.argsort(get_ranked_indices(features_array,labels_array,rank_metric,n_top_feats))]
            print sorted_features.shape
            print 'first instance', sorted_features[0]


            # print 'first item in sorted features \n',sorted_features[0]


            print features_array.shape
            print sorted_features.shape



            if dataset == 'arcene' and sorted_features.shape[1] >9000:
                sorted_features,number_of_features = cut_off_arcene(sorted_features)


            number_of_samples = sorted_features.shape[0]
            scaler = preprocessing.StandardScaler().fit(sorted_features)
            sorted_features = scaler.transform(sorted_features, number_of_samples)
            sorted_features = preprocessing.scale(sorted_features, axis=1)


            sorted_features,number_of_features = discard_bottom_n(sorted_features,number_remaining_feats)

            print sorted_features[0]
            print number_of_features



        print "\n\n "

        ###### THIS BIT TO GET NORMAL AND PRIVILEGED FEATURES ##########

        normal_indices = np.arange(0, n_top_feats)
        print '\n\n\n NORMAL INDICES', normal_indices
        remaining_indices = [index for index in range(number_of_features) if index not in normal_indices]
        if len(remaining_indices) > 1:
            number_of_indices_to_take = len(remaining_indices)/prop_priv
            print "number_of_indices_to_take:", number_of_indices_to_take, "of", len(remaining_indices)
            remaining_indices = remaining_indices[:number_of_indices_to_take]
        else:
            print "1 or less indices remaining. Took ",len(remaining_indices)


        normal_features = sorted_features[:, normal_indices]
        privileged_features = sorted_features[:, remaining_indices]

        print 'all features first item', sorted_features[0]
        print 'normal_features first item', normal_features[0]
        print 'priv features first item', privileged_features[0]
        # sys.exit(0)

        SVM_score = []
        LUPI_score = []


        skf = StratifiedKFold(labels_array, num_folds)

        k=-1

        for train, test in skf:
            k+=1
            param_estimation_file.write("\n\n\n n="+str(n_top_feats)+" fold = "+str(k))


            print "\n \n Top ", n_top_feats ," features; Fold number ", k

            normal_training_data, normal_testing_data = normal_features[train], normal_features[test]



            privileged_training_data, privileged_testing_data = privileged_features[train], privileged_features[test]
            all_training_data, all_testing_data = sorted_features[train], sorted_features[test]

            training_labels, testing_labels = labels_array[train], labels_array[test]


            number_of_items  = (sorted_features.shape[0])
            number_of_training_instances = int(number_of_items -  (number_of_items/num_folds))-1
            # print number_of_training_instances

            rs = ShuffleSplit((number_of_training_instances-1), n_iter=num_folds, test_size=.2, random_state=0)


            if n_top_feats == tuple[0]:

                print "first iteration, doing param selection for baseline"

                ############################### PARAM EST FOR BASELINE
                print 'param est for baseline'
                param_estimation_file.write("\n\n Baseline parameter selection \n C,gamma,score")
                if peeking == True:
                    best_C_baseline, best_gamma_baseline = param_estimation( param_estimation_file, all_training_data, training_labels, c_values,
                    rs, False, None, peeking = True, testing_features=all_testing_data, testing_labels = testing_labels)
                else:
                    best_C_baseline, best_gamma_baseline = param_estimation( param_estimation_file, all_training_data, training_labels, c_values,
                    rs, False, None)

                            # ## Baseline  ###

                clf = svm.SVC(C=best_C_baseline, kernel='rbf', gamma = best_gamma_baseline)
                clf.fit(all_training_data, training_labels)



                baseline_predictions = clf.predict(all_testing_data)
                baseline_score.append(f1_score(testing_labels, baseline_predictions))




            ############################### PARAM EST FOR SVM

            param_estimation_file.write("\n\n SVM parameter selection for top "+str(n_top_feats)+" features\n"+"C,gamma,score")

            if peeking == True:
                best_C_SVM, best_gamma_SVM = param_estimation (param_estimation_file,  normal_training_data, training_labels, c_values,
                rs, privileged=False, privileged_training_data=None, peeking=True, testing_features=normal_testing_data, testing_labels=testing_labels)
            else:
                best_C_SVM, best_gamma_SVM = param_estimation(param_estimation_file, normal_training_data, training_labels, c_values, rs,
                privileged=False, privileged_training_data=None)


            ############# PARAM EST FOR SVM+ ################


            if n_top_feats != number_of_features:

                param_estimation_file.write("\n\n SVM+ parameter selection for top "+str(n_top_feats)+" features\n"+"C,gamma,C*,gamma*,score")
                if peeking == True:
                    best_C_SVM_plus, best_gamma_SVM_plus, best_C_star_SVM_plus, best_gamma_star_SVM_plus = param_estimation(param_estimation_file,
                    normal_training_data, training_labels, c_values, rs, privileged=True, privileged_training_data=privileged_training_data,
                    peeking=True, testing_features=normal_testing_data, testing_labels=testing_labels, multiplier=multiplier)



                else:

                    best_C_SVM_plus, best_gamma_SVM_plus, best_C_star_SVM_plus, best_gamma_star_SVM_plus = param_estimation(param_estimation_file,
                    normal_training_data,  training_labels, c_values, rs, privileged = True, privileged_training_data = privileged_training_data)

                    # code below is for fixing C* and gamma* as the same as C and gamma (param estimation module also needs to be changed)

                    # best_C_SVM_plus, best_gamma_SVM_plus = param_estimation(param_estimation_file,
                    # normal_training_data,  training_labels, c_values, rs, privileged=True, privileged_training_data=privileged_training_data)
                    # best_C_star_SVM_plus, best_gamma_star_SVM_plus = best_C_SVM_plus, best_gamma_SVM_plus




            print "best baseline params",best_C_baseline, best_gamma_baseline
            print "best SVM params", best_C_SVM, best_gamma_SVM
            print "best SVM+ params", best_C_SVM_plus, best_gamma_SVM_plus, best_C_star_SVM_plus, best_gamma_star_SVM_plus

            chosen_params_file.write("\n\n"+str(n_top_feats)+" top features,fold "+str(k)+",baseline,"+str(best_C_baseline)+","+str(best_gamma_baseline))
            chosen_params_file.write("\n ,,SVM,"+str(best_C_SVM)+","+str(best_gamma_SVM))
            chosen_params_file.write("\n  ,,SVM+,"+str(best_C_SVM_plus)+","+str(best_gamma_SVM_plus)+","+str(best_C_star_SVM_plus)+","+str(best_gamma_star_SVM_plus))

            # ###############

            # ## normal classifier  ###


            clf = svm.SVC(C=best_C_SVM, kernel='rbf', gamma = best_gamma_SVM)  # , gamma=gamma)
            clf.fit(normal_training_data, training_labels)
            SVM_predictions = clf.predict(normal_testing_data)
            SVM_score.append(f1_score(testing_labels,SVM_predictions))



            # ## SVM+ classifier  ###
            print "ntopfeats",n_top_feats
            if n_top_feats != total_num_feats:
                print 'doing SVM plus'

                # alphas, bias = svmplusQP(normal_training_data, training_labels.ravel(),
                #                          privileged_training_data, best_C_SVM, best_C_SVM, best_gamma_SVM, best_gamma_SVM)
                alphas, bias = svmplusQP(normal_training_data, training_labels .ravel(),
                                         privileged_training_data, best_C_SVM_plus, best_C_star_SVM_plus, best_gamma_SVM_plus, best_gamma_star_SVM_plus)

                LUPI_predictions_for_testing = svmplusQP_Predict(normal_training_data, normal_testing_data,
                                                                 alphas, bias).ravel()

                LUPI_score.append(f1_score(testing_labels, LUPI_predictions_for_testing))




        results.append(SVM_score)  # put score and feature rank into an array
        if n_top_feats != total_num_feats:
            LUPI_results.append(LUPI_score)
        numbers_of_features_list.append(n_top_feats)


    # print "baseline_score ", baseline_score
    baseline_results = [baseline_score]* len(numbers_of_features_list)


    keyword = str(dataset)+"  (" +str(number_of_items)+ "x" + str(number_of_features) + ");\n peeking =" +str(peeking)\
              +" ; "+str(num_folds)+" folds; feat ranking by "+str(rank_metric)+"; top 1/"+str(prop_priv)+" unselected used as privileged"

    print 'LUPI LENGTH', len(LUPI_results)
    print 'normal length', len(results)
    print 'num feats list', len(numbers_of_features_list)
    get_figures(numbers_of_features_list, results, LUPI_results, baseline_results, num_folds, output_directory, keyword, bottom_n_percent)


    results_file.write(str(np.mean(baseline_results, axis=1)))
    results_file.write(str(np.mean(results, axis=1)))
    results_file.write(str(np.mean(LUPI_results, axis=1)))


def cut_off_arcene(sorted_features):
    sorted_features = sorted_features[:, :9920]
    print 'using arcene dataset: discarding bottom 80 features that consist of 0s'
    print sorted_features.shape
    number_of_features = 9920
    return sorted_features, number_of_features

def discard_bottom_n(sorted_features, number_remaining_feats):


    print sorted_features.shape
    sorted_features = sorted_features[:,:number_remaining_feats]
    print sorted_features.shape
    return sorted_features, sorted_features.shape[1]
