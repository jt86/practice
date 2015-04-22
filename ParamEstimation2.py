__author__ = 'jt306'
import numpy as np
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.metrics import f1_score, pairwise
from sklearn import svm
from sklearn import grid_search
import logging
from Get_Full_Path import get_full_path


test_folds_file = open(get_full_path('Desktop/Privileged_Data/test_folds_file2.csv'),'r+')


def get_gamma_from_c(c_values, features):
    euclidean_distance = pairwise.euclidean_distances(features)
    median_euclidean_distance = np.median(euclidean_distance ** 2)
    return [value / median_euclidean_distance for value in c_values]


def param_estimation(param_estimation_file, training_features, training_labels, c_values, rs, privileged,
                     privileged_training_data=None, peeking=False, testing_features=None, testing_labels=None,
                     multiplier=1, logger=None):
    training_labels=training_labels.ravel()

    test_folds_file.write("VALUES FROM PARAM ESTIMATION")



    logger.info('Selecting hyperparameter C from values: %r', c_values)
    # logger.info('Selecting hyperparameter gamma from values: %r', gamma_values)

    dict_of_parameters = {}
    if privileged == True:
        code_with_score = np.zeros((len(c_values) ** 2))
    else:
        code_with_score = np.zeros(len(c_values))

    if peeking == True:
        # logging.info( 'doing param selection by peeking at test data')

        get_scores_for_this_fold(privileged, c_values, dict_of_parameters, training_features, training_labels,
                                 testing_features, testing_labels, privileged_training_data, code_with_score)

        output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)
        best_parameters = dict_of_parameters[code_with_score.argmax(axis=0)]
        return best_parameters

    else:

        nested_fold = 0
        for train_indices, test_indices in rs:

            test_folds_file.write("\n train \n")
            test_folds_file.write(str(train_indices))
            test_folds_file.write("\n test \n")
            test_folds_file.write(str(test_indices))


            # print 'test indices ',nested_fold
            # print test_indices

            nested_fold += 1
            # logging.info( "length of training features", len(training_features))

            train_this_fold, test_this_fold = training_features[train_indices], training_features[test_indices]
            train_labels_this_fold, test_labels_this_fold = training_labels[train_indices], training_labels[test_indices]

            if privileged == True:
                privileged_train_this_fold = privileged_training_data[train_indices]
            else:
                privileged_train_this_fold = None
            get_scores_for_this_fold(privileged, c_values, dict_of_parameters, train_this_fold, train_labels_this_fold, test_this_fold, test_labels_this_fold, privileged_train_this_fold, code_with_score)



        best_parameters = dict_of_parameters[code_with_score.argmax(axis=0)]
        # output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)
        logging.info('best parameters %r', best_parameters)
        return best_parameters



def output_params_with_scores(dictionary, code_with_score, param_estimation_file):
    for index in range(len(code_with_score)):
        settings_with_score = str(dictionary[index]).translate(None, '[]')

        param_estimation_file.write("\n" + settings_with_score + ",")
        param_estimation_file.write(str(code_with_score[index]))



def get_scores_for_this_fold(privileged,c_values,dict_of_parameters, train_data, train_labels, test_data, test_labels, priv_train, code_with_score):
    j = 0

    if privileged == False:
        for c_value in c_values:
            #for gamma_value in gamma_values:
            dict_of_parameters[j] = [c_value]#, gamma_value]
            clf = svm.SVC(C=c_value, kernel='linear')#, gamma=gamma_value)
            clf.fit(train_data, train_labels)
            new_score = f1_score(test_labels, clf.predict(test_data))
            # logging.info( "new score:",new_score)
            code_with_score[j] += new_score
            j += 1

    if privileged == True:
        # gamma_values = [value * multiplier for value in gamma_values]
        # gamma_star_values = [value * multiplier for value in
        #                      get_gamma_from_c(c_values, privileged_train_this_fold)]
        for c_value in c_values:
            # for gamma_value in gamma_values:
            for c_star_value in c_values:
                    # for gamma_star_value in gamma_star_values:
                    dict_of_parameters[j] = [c_value, c_star_value]
                    alphas, bias = svmplusQP(X=train_data, Y=train_labels,
                                             Xstar=priv_train,
                                             C=c_value, Cstar=c_value) #gamma=gamma_value,
                                             #gammastar=gamma_value)

                    predictions_this_fold = svmplusQP_Predict(train_data, test_data, alphas, bias)
                    new_score = f1_score(test_labels, predictions_this_fold)
                    # logging.info( "new score:",new_score)
                    code_with_score[j] += new_score

                    j += 1

