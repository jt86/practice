__author__ = 'jt306'
import numpy as np
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.metrics import f1_score, pairwise
from sklearn import svm
from sklearn import grid_search
import logging


def get_gamma_from_c(c_values, features):
    euclidean_distance = pairwise.euclidean_distances(features)
    median_euclidean_distance = np.median(euclidean_distance ** 2)
    return [value / median_euclidean_distance for value in c_values]


def param_estimation(param_estimation_file, training_features, training_labels, c_values, rs, privileged,
                     privileged_training_data=None, peeking=False, testing_features=None, testing_labels=None,
                     multiplier=1, logger=None):
    training_labels=training_labels.ravel()
    # logging.info( "training_features[0]:")
    # logging.info( training_features[0:5, :6])
    # gamma_values = get_gamma_from_c(c_values, training_features)

    logger.info('Selecting hyperparameter C from values: %r', c_values)
    # logger.info('Selecting hyperparameter gamma from values: %r', gamma_values)


    if peeking == True:
        # logging.info( 'doing param selection by peeking at test data')
        return peeking_param_estimation(param_estimation_file, training_features, training_labels, c_values, privileged,
                                        privileged_training_data, testing_features, testing_labels, multiplier)

    else:

        dict_of_parameters = {}

        if privileged == True:
            code_with_score = np.zeros((len(c_values) ** 4))
        else:
            code_with_score = np.zeros(len(c_values) ** 2)

        nested_fold = 0
        for train_indices, test_indices in rs:

            nested_fold += 1
            # logging.info( "length of training features", len(training_features))


            train_this_fold = training_features[train_indices]
            test_this_fold = training_features[test_indices]

            train_labels_this_fold = training_labels[train_indices]
            test_labels_this_fold = training_labels[test_indices]

            j = 0

            if privileged == False:
                for c_value in c_values:
                    #for gamma_value in gamma_values:
                    dict_of_parameters[j] = [c_value]#, gamma_value]
                    clf = svm.SVC(C=c_value, kernel='linear')#, gamma=gamma_value)
                    clf.fit(train_this_fold, train_labels_this_fold)
                    new_score = f1_score(test_labels_this_fold, clf.predict(test_this_fold))
                    # logging.info( "new score:",new_score)
                    code_with_score[j] += new_score
                    j += 1

            if privileged == True:
                privileged_train_this_fold = privileged_training_data[train_indices]
                # gamma_values = [value * multiplier for value in gamma_values]
                # gamma_star_values = [value * multiplier for value in
                #                      get_gamma_from_c(c_values, privileged_train_this_fold)]
                for c_value in c_values:
                    # for gamma_value in gamma_values:
                    for c_star_value in c_values:
                            # for gamma_star_value in gamma_star_values:
                            dict_of_parameters[j] = [c_value, c_star_value]
                            alphas, bias = svmplusQP(X=train_this_fold, Y=train_labels_this_fold,
                                                     Xstar=privileged_train_this_fold,
                                                     C=c_value, Cstar=c_value) #gamma=gamma_value,
                                                     #gammastar=gamma_value)

                            predictions_this_fold = svmplusQP_Predict(train_this_fold, test_this_fold, alphas, bias)
                            new_score = f1_score(test_labels_this_fold, predictions_this_fold)
                            # logging.info( "new score:",new_score)
                            code_with_score[j] += new_score

                            j += 1

        best_parameters = dict_of_parameters[code_with_score.argmax(axis=0)]
        # output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)
        logging.info('best parameters %r', best_parameters)
        return best_parameters


def peeking_param_estimation(param_estimation_file, training_features, training_labels, c_values, privileged,
                             privileged_training_data=None, testing_features=None, testing_labels=None, multiplier=1):
    # gamma_values = get_gamma_from_c(c_values, training_features)
    dictionary0 = {}
    if privileged == True:
        code_with_score = np.zeros((len(c_values) ** 2))
    else:
        code_with_score = np.zeros(len(c_values))

    j = 0

    if privileged == False:
        for c_value in c_values:
            # for gamma_value in gamma_values:
            dictionary0[j] = [c_value]#, gamma_value]
            clf = svm.SVC(C=c_value, kernel='linear')#, gamma=gamma_value)
            clf.fit(training_features, training_labels)

            new_score = f1_score(testing_labels, clf.predict(testing_features))
            # logging.info( "new score:",new_score)
            code_with_score[j] += new_score
            j += 1

    if privileged == True:
        # gamma_values = [value * multiplier for value in gamma_values]
        # gamma_star_values = [value * multiplier for value in get_gamma_from_c(c_values, privileged_training_data)]

        for c_value in c_values:
            # for gamma_value in gamma_values:
            for c_star_value in c_values:
                    # for gamma_star_value in gamma_star_values:
                    dictionary0[j] = [c_value, c_star_value]#, gamma_star_value]

                    alphas, bias = svmplusQP(X=training_features, Y=training_labels, Xstar=privileged_training_data,
                                             C=c_value, Cstar=c_star_value)#, gamma=gamma_value,
                                             #gammastar=gamma_star_value)
                    predictions_this_fold = svmplusQP_Predict(training_features, testing_features, alphas, bias)


                    new_score = f1_score(testing_labels, predictions_this_fold)
                    # logging.info( "new score:",new_score)
                    code_with_score[j] += new_score

                    j += 1

    output_params_with_scores(dictionary0, code_with_score, param_estimation_file)
    best_parameters = dictionary0[code_with_score.argmax(axis=0)]
    return best_parameters


def output_params_with_scores(dictionary, code_with_score, param_estimation_file):
    for index in range(len(code_with_score)):
        settings_with_score = str(dictionary[index]).translate(None, '[]')

        param_estimation_file.write("\n" + settings_with_score + ",")
        param_estimation_file.write(str(code_with_score[index]))



