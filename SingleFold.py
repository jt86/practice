import os, sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy, pairwise
from sklearn import svm, linear_model
from SVMplus3 import svmplusQP, svmplusQP_Predict
from ParamEstimation2 import param_estimation
# from MainFunctionParallelised import get_indices_for_fold, get_train_test_selected_unselected, get_percentage_of_t, get_sorted_features
import sklearn.preprocessing
from InitialFeatSelection import get_best_feats
import sklearn.preprocessing as preprocessing
from FeatSelection import get_ranked_indices, recursive_elimination2
from GetFeatsAndLabels import get_feats_and_labels
import argparse
from Get_Full_Path import get_full_path
from Get_Awa_Data import get_awa_data





def single_fold(k, num_folds, take_t, bottom_n_percent,
         rank_metric, dataset, peeking, kernel,
         cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):

        c_values, cstar_values = get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin, cstarmax)

        all_results_directory = get_full_path('Desktop/Privileged_Data/FixedCandCStar/{}'.format(dataset))
        if not os.path.exists(all_results_directory):
            os.mkdir(all_results_directory)
        output_directory = all_results_directory

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        original_features_array, labels_array, tuple = get_feats_and_labels(dataset)
        param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
        chosen_params_file = open(os.path.join(output_directory, 'chosen_parameters.csv'), "a")

        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        if not os.path.exists(cross_validation_folder):
            os.mkdir(cross_validation_folder)

        list_of_t = []
        inner_folds = num_folds


        if k==0:
            # with open(os.path.join(cross_validation_folder,'keyword.txt'),'a') as keyword_file:
            #     keyword_file.write("{} ({}x{}) t values:{}\n peeking={}; {} folds; metric: {}; c={{10^{}..10^{}}}; c*={{10^{}..10^{}}} ({} values)".format(dataset,
            #            original_features_array.shape[0], original_features_array.shape[1], list_of_t, peeking, num_folds, rank_metric, cmin, cmax, cstarmin, cstarmax, number_of_cs))

            with open(os.path.join(cross_validation_folder,'keyword.txt'),'a') as keyword_file:
                keyword_file.write("{} t values:{}\n peeking={}; {} folds; metric: {}; c={{10^{}..10^{}}}; c*={{10^{}..10^{}}} ({} values)".format(dataset,
                        list_of_t, peeking, num_folds, rank_metric, cmin, cmax, cstarmin, cstarmax, number_of_cs))

        if 'awa' in dataset:
            print 'last symbol', dataset[-1]
            all_training, all_testing, training_labels, testing_labels = get_awa_data("", dataset[-1])
            original_number_feats = all_training.shape[1]
            top_t_training =all_training
            top_t_testing = all_testing
            total_number_of_items = top_t_training.shape[0]

        else:

            original_number_feats = original_features_array.shape[1]



            print'k',k
            train, test = get_indices_for_fold(labels_array, num_folds, k)
            top_t_training,top_t_testing, unselected_features_training, unselected_features_testing = \
                get_train_test_selected_unselected(k, labels_array, original_features_array, c_values, num_folds, take_t, train, test)


            total_number_of_items = (train.shape[0])
            number_of_training_instances = int(total_number_of_items - (total_number_of_items / num_folds)) - 1

            all_training, all_testing = original_features_array[train], original_features_array[test]
            training_labels, testing_labels = labels_array[train], labels_array[test]


            ######################

        t = top_t_training.shape[1]
        list_of_t.append(t)
        number_remaining_feats = original_number_feats - (bottom_n_percent * (original_number_feats) / 100)
            #######################

            # list_of_values = get_percentage_of_t(t)[0]

        numbers_of_features_list=[]

        # list_of_values = get_percentage_of_t(t)[0]
        # if take_t:
        #     list_of_values = get_percentage_of_t(t)[0]
        # else:
        #     list_of_values = [i for i in range(*tuple)]
        #
        #
        # list_of_percentages = get_percentage_of_t(t)[1]

        list_of_values = []
        for percentage in [5,10,25,50,75]:
            list_of_values.append(t*percentage/100)

        for n_top_feats in list_of_values:

            # if n_top_feats not in numbers_of_features_list:
            #     numbers_of_features_list.append(n_top_feats)

            param_estimation_file.write("\n\n n={},fold={}".format(n_top_feats,k))

            # normal_indices = np.arange(0, n_top_feats)
            # privileged_indices = [index for index in range(t) if index not in normal_indices]
            #

            best_n_mask = recursive_elimination2(all_training, training_labels, n_top_feats)
            normal_features_training = all_training[:,best_n_mask]
            normal_features_testing = all_testing[:,best_n_mask]
            privileged_features_training = all_training[:, np.invert(best_n_mask)]

            #  # Get T, Get NORMAL and PRIVILEGED
            # print 'top t training', top_t_training.shape
            # print 'training labels', training_labels.shape
            #
            # top_t_sorted_training = get_sorted_features(top_t_training, training_labels, rank_metric, n_top_feats, dataset,  number_remaining_feats)
            # top_t_sorted_testing = get_sorted_features(top_t_testing, testing_labels, rank_metric, n_top_feats, dataset,  number_remaining_feats)
            #
            # normal_indices = np.arange(0, n_top_feats)
            # print 'n top feats',n_top_feats
            # privileged_indices = [index for index in range(t) if index not in normal_indices]
            # print 'normal indices', normal_indices
            # if len(normal_indices) == 0:
            #     normal_indices = [0]
            # print len(normal_indices)
            # print normal_indices
            #
            # normal_features_training = top_t_sorted_training[:,normal_indices]
            # normal_features_testing = top_t_sorted_testing[:,normal_indices]
            # privileged_features_training = top_t_sorted_training[:, privileged_indices]
            #
            # if take_t:
            #     privileged_features_training = np.hstack([privileged_features_training,unselected_features_training])                     #todo uncomment this!
            #     # privileged_features_training = privileged_features_training[:(privileged_features_training.shape[1]/prop_priv)]
            #
            #
            #
            # number_of_training_instances = int(total_number_of_items - (total_number_of_items / num_folds)) - 1
            # ##############################  BASELINE - all features



            if n_top_feats == list_of_values[0]:

                param_estimation_file.write("\n\n Baseline scores array")

                rs = StratifiedShuffleSplit(y=training_labels, n_iter=inner_folds, test_size=.2, random_state=0)
                best_C_baseline = param_estimation(param_estimation_file, all_training,
                                                                        training_labels, c_values,rs, False, None,
                                                                        peeking,testing_features=all_testing,
                                                                        testing_labels=testing_labels)


                print 'best c baseline',best_C_baseline,  'kernel', kernel
                clf = svm.SVC(C=best_C_baseline, kernel=kernel)
                # pdb.set_trace()
                print all_training.shape, training_labels.shape
                clf.fit(all_training, training_labels)

                baseline_predictions = clf.predict(all_testing)


                with open(os.path.join(cross_validation_folder,'baseline.csv'),'a') as cv_baseline_file:
                    cv_baseline_file.write(str(accuracy(testing_labels, baseline_predictions))+",")


                # ##############################  BASELINE 2 - top t features only

                # rs = ShuffleSplit((number_of_training_instances - 1), n_iter=num_folds, test_size=.2, random_state=0)
                rs = StratifiedShuffleSplit(y=training_labels, n_iter=inner_folds, test_size=.2, random_state=0)
                best_C_baseline2 = param_estimation(param_estimation_file, top_t_training,
                                                                        training_labels, c_values,rs, False, None,
                                                                        peeking,testing_features=top_t_testing,
                                                                        testing_labels=testing_labels)

                print 'best c baseline2',best_C_baseline2
                clf2 = svm.SVC(C=best_C_baseline2, kernel=kernel)
                clf2.fit(top_t_training, training_labels)

                baseline_predictions2 = clf2.predict(top_t_testing)

                with open(os.path.join(cross_validation_folder,'baseline2.csv'),'a') as cv_baseline_file2:
                    cv_baseline_file2.write(str(accuracy(testing_labels, baseline_predictions2))+",")



            ############################### SVM - PARAM ESTIMATION AND RUNNING
            total_number_of_items = top_t_training.shape[0]
            # rs = ShuffleSplit((number_of_training_instances - 1), n_iter=num_folds, test_size=.2, random_state=0)
            rs = StratifiedShuffleSplit(y=training_labels, n_iter=inner_folds, test_size=.2, random_state=0)
            param_estimation_file.write(
                # "\n\n SVM parameter selection for top " + str(n_top_feats) + " features\n" + "C,score")
                "\n\n SVM scores array for top " + str(n_top_feats) + " features\n")


            best_C_SVM  = param_estimation(param_estimation_file, normal_features_training,
                                          training_labels, c_values, rs, privileged=False, privileged_training_data=None,
                                        peeking=peeking, testing_features=normal_features_testing,testing_labels=testing_labels)

            clf = svm.SVC(C=best_C_SVM, kernel=kernel)
            clf.fit(normal_features_training, training_labels)
            with open(os.path.join(cross_validation_folder,'svm-{}.csv'.format(k)),'a') as cv_svm_file:
                cv_svm_file.write(str(accuracy(testing_labels, clf.predict(normal_features_testing)))+",")


            ############# SVM PLUS - PARAM ESTIMATION AND RUNNING
            # rs = ShuffleSplit((number_of_training_instances - 1), n_iter=10, test_size=.2, random_state=0)
            rs = StratifiedShuffleSplit(y=training_labels, n_iter=inner_folds, test_size=.2, random_state=0)
            if n_top_feats != number_remaining_feats:
                assert n_top_feats < number_remaining_feats
                param_estimation_file.write(
                # "\n\n SVM PLUS parameter selection for top " + str(n_top_feats) + " features\n" + "C,C*,score")
                "\n\n SVM PLUS scores array for top " + str(n_top_feats) + " features\n")

                # best_C_SVM_plus,  best_C_star_SVM_plus   = param_estimation(
                #     param_estimation_file, normal_features_training, training_labels, c_values, rs, privileged=True,
                #     privileged_training_data=privileged_features_training,peeking=peeking,
                #     testing_features=normal_features_testing, testing_labels=testing_labels,
                #     cstar_values=cstar_values)
                best_C_SVM_plus,  best_C_star_SVM_plus = 1, 100

                alphas, bias = svmplusQP(normal_features_training, training_labels.ravel(), privileged_features_training,
                                         best_C_SVM_plus, best_C_star_SVM_plus)


                LUPI_predictions_for_testing = svmplusQP_Predict(normal_features_training, normal_features_testing,
                                                                 alphas, bias, kernel).ravel()


                with open(os.path.join(cross_validation_folder,'lupi-{}.csv'.format(k)),'a') as cv_lupi_file:
                    cv_lupi_file.write(str(accuracy(testing_labels, LUPI_predictions_for_testing))+",")

            chosen_params_file.write("\n\n{} top features,fold {},baseline,{}".format(n_top_feats,k,best_C_baseline))
            chosen_params_file.write("\n,,SVM,{}".format(best_C_SVM))
            chosen_params_file.write("\n,,SVM+,{},{}" .format(best_C_SVM_plus,best_C_star_SVM_plus))



def cut_off_arcene(sorted_features):
    sorted_features = sorted_features[:, :9920]

    return sorted_features


def discard_bottom_n(sorted_features, number_remaining_feats):
    sorted_features = sorted_features[:, :number_remaining_feats]
    return sorted_features

def take_top_feats(privileged_indices,prop_priv):
    if len(privileged_indices) > 1:
        number_of_indices_to_take = len(privileged_indices) / prop_priv

        privileged_indices = privileged_indices[:number_of_indices_to_take]
    else:

        return privileged_indices

def get_sorted_features(features_array, labels_array, rank_metric, n_top_feats,dataset, number_remaining_feats):
    sorted_features = features_array[:,np.argsort(get_ranked_indices(features_array, labels_array, rank_metric, n_top_feats))]



    if dataset == 'arcene' and sorted_features.shape[1] > 9000:
        sorted_features = cut_off_arcene(sorted_features)

    number_of_samples = sorted_features.shape[0]
    # scaler = preprocessing.StandardScaler().fit(sorted_features)
    # sorted_features = scaler.transform(sorted_features, number_of_samples)
    # sorted_features = preprocessing.scale(sorted_features, axis=1)
    normaliser = preprocessing.Normalizer('l1')
    normaliser.fit_transform(sorted_features)
    sorted_features = discard_bottom_n(sorted_features, number_remaining_feats)


    return sorted_features


def get_percentage_of_t(t, tuple=(10,101,15)):
    list_of_values, list_of_percentages =[],[]
    for i in range(*tuple):
        print i
        list_of_values.append(t*i/100)
        list_of_percentages.append(i)
    return list_of_values,list_of_percentages


def get_training_testing(take_t,all_training,all_testing,training_labels,c_values, num_folds,number_of_training_instances):
    if take_t == True:
        print 'taking top t only'
        rs = ShuffleSplit((number_of_training_instances - 1), n_iter=10, test_size=.2, random_state=0)
        top_t_indices, remaining_indices = get_best_feats(all_training,training_labels,c_values, num_folds, rs, 'heart')
        top_t_training, unselected_features_training = all_training[:,top_t_indices], all_training[:,remaining_indices]
        top_t_testing, unselected_features_testing = all_testing[:,top_t_indices], all_testing[:,remaining_indices]

    else:
        top_t_training = all_training
        top_t_testing = all_testing
        unselected_features_training, unselected_features_testing = None,None

    return top_t_training,top_t_testing, unselected_features_training, unselected_features_testing



def get_indices_for_fold(labels_array, num_folds, fold_num):
    for index, (train,test) in enumerate(StratifiedKFold(labels_array, num_folds, shuffle=False , random_state=1)):
        if index==fold_num:
            return train,test


def get_train_test_selected_unselected(k, labels_array, original_features_array, c_values, num_folds, take_t, train, test):

    number_of_training_instances = int(len(train) - (len(train) / num_folds)) - 1
    print 'number_of_training_instances', number_of_training_instances
    print train,test
    all_training, all_testing = original_features_array[train], original_features_array[test]
    training_labels, testing_labels = labels_array[train], labels_array[test]
    return get_training_testing(take_t,all_training,all_testing,training_labels,c_values, num_folds,number_of_training_instances)

def get_c_and_cstar(cmin,cmax,number_of_cs, cstarmin=None, cstarmax=None):
    c_values = np.logspace(cmin,cmax,number_of_cs)
    if cstarmin==None:
        cstarmin, cstarmax = cmin,cmax
    cstar_values=np.logspace(cstarmin,cstarmax,number_of_cs)
    return c_values, cstar_values

#
# with open('/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/test-parallel/wine_peeking=True_folds=3_metric=r2_c=10^-1-10^1_cstar=10^None-10^None/cross-validation/parameters_file','r') as parameters_file:
#     first_setting = (parameters_file.readline())
#     single_fold(first_setting)
#     print first_setting

# single_fold(k=3, num_folds=5, take_t=False, bottom_n_percent=0, rank_metric='r2', dataset='wine', peeking=True, kernel='rbf', cmin=0.1, cmax=10., number_of_cs=1)


# single_fold(k=4, num_folds=5, take_t=False, bottom_n_percent=0, rank_metric='r2', dataset='awa1', peeking=True, kernel='linear', cmin=0.1, cmax=10., number_of_cs=3)