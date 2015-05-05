__author__ = 'jt306'
import numpy as np
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.metrics import f1_score, pairwise
from sklearn import svm
from sklearn import grid_search
import logging
from Get_Full_Path import get_full_path


def get_gamma_from_c(c_values, features):
    # print 'getgamma c_values', c_values
    euclidean_distance = pairwise.euclidean_distances(features)
    median_euclidean_distance = np.median(euclidean_distance ** 2)
    return [value / median_euclidean_distance for value in c_values]


def param_estimation(param_estimation_file, training_features, training_labels, c_values, rs, privileged,
                     privileged_training_data=None, peeking=False, testing_features=None, testing_labels=None,
                     cstar_values=None):
    training_labels=training_labels.ravel()

    gamma_values=  get_gamma_from_c(c_values, training_features)
    gammastar_values=None

    kernel = 'linear'
    if kernel == 'linear':
        if privileged:
            scores_array = np.zeros((len(c_values),len(cstar_values)))
        else:
            scores_array = np.zeros(len(c_values))
    if kernel == 'rbf':
        if privileged:
            gammastar_values=  get_gamma_from_c(cstar_values, training_features)
            scores_array = np.zeros((len(c_values),len(cstar_values),len(gamma_values),len(gammastar_values)))
        else:
            scores_array = np.zeros((len(c_values),len(gamma_values)))


    if peeking == True:

        scores_array = get_scores_for_this_fold(privileged, c_values, training_features, training_labels,
                                 testing_features, testing_labels, privileged_training_data, cstar_values, scores_array, gamma_values=gamma_values, gammastar_values=gammastar_values, kernel=kernel)

        # output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)


    else:

        print 'scores array \n', scores_array

        for train_indices, test_indices in rs:
            train_this_fold, test_this_fold = training_features[train_indices], training_features[test_indices]
            train_labels_this_fold, test_labels_this_fold = training_labels[train_indices], training_labels[test_indices]
            assert train_labels_this_fold.shape[0]+test_labels_this_fold.shape[0]==training_labels.shape[0], 'number of labels total'

            if privileged == True:
                privileged_train_this_fold = privileged_training_data[train_indices]
            else:
                privileged_train_this_fold = None

            scores_array = get_scores_for_this_fold(privileged, c_values, train_this_fold, train_labels_this_fold,
                                                    test_this_fold, test_labels_this_fold, privileged_train_this_fold, cstar_values, scores_array, gamma_values=gamma_values, gammastar_values=gammastar_values, kernel=kernel)


    best_indices = np.unravel_index(scores_array.argmax(), scores_array.shape)

    if kernel == 'linear':
        if privileged:
            best_parameters = c_values[best_indices[0]],cstar_values[best_indices[1]]
        else:
            best_parameters = c_values[best_indices[0]]

    if kernel == 'rbf':
        if privileged:
            best_parameters = c_values[best_indices[0]],cstar_values[best_indices[1]],gamma_values[best_indices[2]],gammastar_values[best_indices[3]]
        else:
            best_parameters = c_values[best_indices[0]],gamma_values[best_indices[1]]


    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    param_estimation_file.write(np.array2string(scores_array, separator=', ').translate(None, '[]'))
    return best_parameters



def output_params_with_scores(dictionary, code_with_score, param_estimation_file):
    for index in range(len(code_with_score)):
        settings_with_score = str(dictionary[index]).translate(None, '[]')

        param_estimation_file.write("\n" + settings_with_score + ",")
        param_estimation_file.write(str(code_with_score[index]))
        #
        # param_estimation_file.write('\n')
        # param_estimation_file.write(','.join(map(str,code_with_score)))




def get_scores_for_this_fold(privileged,c_values,train_data, train_labels, test_data, test_labels, priv_train, cstar_values, scores_array,gamma_values, gammastar_values,kernel):
    if kernel == 'linear':
        if privileged == False:
            for c_index, c_value in enumerate(c_values):
                clf = svm.SVC(C=c_value, kernel='linear')
                clf.fit(train_data, train_labels)
                scores_array[c_index]+=f1_score(test_labels, clf.predict(test_data))
        if privileged == True:
            for c_index, c_value in enumerate(c_values):
                for cstar_index, cstar_value in enumerate(cstar_values):
                        alphas, bias = svmplusQP(X=train_data, Y=train_labels,
                                                 Xstar=priv_train,
                                                 C=c_value, Cstar=cstar_value)
                        predictions_this_fold = svmplusQP_Predict(train_data, test_data, alphas, bias)
                        scores_array[c_index,cstar_index]+= f1_score(test_labels, predictions_this_fold)

    if kernel == 'rbf':
        if privileged == False:
            for c_index, c_value in enumerate(c_values):
                for gamma_index, gamma_value in enumerate(gamma_values):
                    clf = svm.SVC(C=c_value, kernel='rbf', gamma=gamma_value)
                    clf.fit(train_data, train_labels)
                    scores_array[c_index, gamma_index]+=f1_score(test_labels, clf.predict(test_data))
        if privileged == True:
            for c_index, c_value in enumerate(c_values):
                for cstar_index, cstar_value in enumerate(cstar_values):
                    for gamma_index, gamma_value in enumerate(gamma_values):
                        for gammastar_index, gammastar_value in enumerate(gammastar_values):
                            alphas, bias = svmplusQP(X=train_data, Y=train_labels,
                                                     Xstar=priv_train,
                                                     C=c_value, Cstar=cstar_value, gamma=gamma_value, gammastar = gammastar_value)
                            predictions_this_fold = svmplusQP_Predict(train_data, test_data, alphas, bias, kernel)
                            # print 'c index', c_index, 'cstar index', cstar_index, 'gamma index', gamma_index, 'gammastarindex',gammastar_index
                            scores_array[c_index,cstar_index,gamma_index,gammastar_index]+= f1_score(test_labels, predictions_this_fold)

    return scores_array
#
#
# def get_scores_for_this_fold(privileged,c_values,dict_of_parameters, train_data, train_labels, test_data, test_labels, priv_train, code_with_score, cstar_values, scores_array):
#
#     j = 0
#
#
#     if privileged == False:
#         for c_index, c_value in enumerate(c_values):
#             dict_of_parameters[j] = [c_value]
#             clf = svm.SVC(C=c_value, kernel='linear')
#             clf.fit(train_data, train_labels)
#             code_with_score[j] += f1_score(test_labels, clf.predict(test_data))
#             j += 1
#             scores_array[c_index]+=f1_score(test_labels, clf.predict(test_data))
#
#
#     if privileged == True:
#         for c_index, c_value in enumerate(c_values):
#             for cstar_index, cstar_value in enumerate(cstar_values):
#                     dict_of_parameters[j] = [c_value, cstar_value]
#                     alphas, bias = svmplusQP(X=train_data, Y=train_labels,
#                                              Xstar=priv_train,
#                                              C=c_value, Cstar=cstar_value)
#                     predictions_this_fold = svmplusQP_Predict(train_data, test_data, alphas, bias)
#                     code_with_score[j] += f1_score(test_labels, predictions_this_fold)
#                     j += 1
#                     scores_array[c_index,cstar_index]+= f1_score(test_labels, predictions_this_fold)
#
#     return scores_array