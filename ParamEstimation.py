__author__ = 'jt306'
import numpy as np
import numpy
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.metrics import accuracy_score, pairwise
from sklearn import svm
from sklearn import grid_search
import logging
from Get_Full_Path import get_full_path
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
import numpy as np
import numpy
import pdb
from sklearn import cross_validation, linear_model
from scipy.optimize import *
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC

def do_CV_svmrfe_5fold(Xorig,Yorig, reg_array, top):

	X = Xorig.copy(); Y = Yorig.copy()
	cv_scores = numpy.zeros(len(reg_array))
	cv = cross_validation.StratifiedKFold(Y, 5)

	for i,(train, test) in enumerate(cv):
		for j, reg_const in enumerate(reg_array):

			svc = SVC(C=reg_const, kernel="linear", random_state=1)
			rfe = RFE(estimator=svc, n_features_to_select=top, step=1)
			rfe.fit(X[train], Y[train])
			cv_scores[j] = cv_scores[j] + rfe.score(X[test], Y[test])
	cv_scores = cv_scores/5.
	reg_best = reg_array[numpy.argmax(cv_scores)]

	return reg_best


def param_estimation(param_estimation_file, training_features, training_labels, c_values, inner_folds):

    training_labels=training_labels.ravel()
    scores_array = np.zeros(len(c_values))
    rs = StratifiedShuffleSplit(y=training_labels, n_iter=inner_folds, test_size=.2, random_state=0)


    for train_indices, test_indices in rs:
        train_this_fold, test_this_fold = training_features[train_indices], training_features[test_indices]
        train_labels_this_fold, test_labels_this_fold = training_labels[train_indices], training_labels[test_indices]


        scores_array = get_scores_for_this_fold(c_values, train_this_fold, train_labels_this_fold,
                                                test_this_fold, test_labels_this_fold, scores_array)

    best_indices = np.unravel_index(scores_array.argmax(), scores_array.shape)
    best_parameters = c_values[best_indices[0]]
    param_estimation_file.write(np.array2string(scores_array, separator=', ').translate(None, '[]'))
    return best_parameters


def get_scores_for_this_fold(c_values,train_data, train_labels, test_data, test_labels, scores_array):

    for c_index, c_value in enumerate(c_values):
        clf = svm.SVC(C=c_value, kernel='linear',random_state=1)
        clf.fit(train_data, train_labels)
        scores_array[c_index]+=accuracy_score(test_labels, clf.predict(test_data))
    return scores_array



#
# def get_gamma_from_c(c_values, features):
#     # print 'getgamma c_values', c_values
#     euclidean_distance = pairwise.euclidean_distances(features)
#     median_euclidean_distance = np.median(euclidean_distance ** 2)
#     return [value / median_euclidean_distance for value in c_values]
#
#
# def param_estimation(param_estimation_file, training_features, training_labels, c_values, inner_folds, privileged,
#                      privileged_training_data=None, peeking=False, testing_features=None, testing_labels=None,
#                      cstar_values=None):
#     training_labels=training_labels.ravel()
#
#     gamma_values=  get_gamma_from_c(c_values, training_features)
#     gammastar_values=None
#
#
#     if privileged:
#         scores_array = np.zeros((len(c_values),len(cstar_values)))
#     else:
#         scores_array = np.zeros(len(c_values))
#
#
#
#     if peeking == True:
#
#         scores_array = get_scores_for_this_fold(privileged, c_values, training_features, training_labels,
#                                  testing_features, testing_labels, privileged_training_data, cstar_values, scores_array, gamma_values=gamma_values, gammastar_values=gammastar_values, kernel=kernel)
#
#         # output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)
#
#
#     else:
#         # rs = ShuffleSplit((number_of_training_instances - 1), n_iter=num_folds, test_size=.2, random_state=0)
#         rs = StratifiedShuffleSplit(y=training_labels, n_iter=inner_folds, test_size=.2, random_state=0)
#         print 'scores array \n', scores_array
#
#         for train_indices, test_indices in rs:
#             train_this_fold, test_this_fold = training_features[train_indices], training_features[test_indices]
#             train_labels_this_fold, test_labels_this_fold = training_labels[train_indices], training_labels[test_indices]
#             assert train_labels_this_fold.shape[0]+test_labels_this_fold.shape[0]==training_labels.shape[0], 'number of labels total'
#
#             if privileged == True:
#                 privileged_train_this_fold = privileged_training_data[train_indices]
#             else:
#                 privileged_train_this_fold = None
#
#             scores_array = get_scores_for_this_fold(privileged, c_values, train_this_fold, train_labels_this_fold,
#                                                     test_this_fold, test_labels_this_fold, privileged_train_this_fold, cstar_values, scores_array, gamma_values=gamma_values, gammastar_values=gammastar_values, kernel=kernel)
#
#
#     best_indices = np.unravel_index(scores_array.argmax(), scores_array.shape)
#
#     if kernel == 'linear':
#         if privileged:
#             best_parameters = c_values[best_indices[0]],cstar_values[best_indices[1]]
#         else:
#             best_parameters = c_values[best_indices[0]]
#
#     if kernel == 'rbf':
#         if privileged:
#             best_parameters = c_values[best_indices[0]],cstar_values[best_indices[1]],gamma_values[best_indices[2]],gammastar_values[best_indices[3]]
#         else:
#             best_parameters = c_values[best_indices[0]],gamma_values[best_indices[1]]
#
#
#     np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#     param_estimation_file.write(np.array2string(scores_array, separator=', ').translate(None, '[]'))
#     return best_parameters
#
#
#
#
#
#
#
# def get_scores_for_this_fold(privileged,c_values,train_data, train_labels, test_data, test_labels, priv_train, cstar_values, scores_array,gamma_values, gammastar_values):
#
#     if privileged == False:
#         for c_index, c_value in enumerate(c_values):
#             clf = svm.SVC(C=c_value, kernel='linear',random_state=1)
#             clf.fit(train_data, train_labels)
#             scores_array[c_index]+=accuracy_score(test_labels, clf.predict(test_data))
#
#     if privileged == True:
#         for c_index, c_value in enumerate(c_values):
#             for cstar_index, cstar_value in enumerate(cstar_values):
#                     alphas, bias = svmplusQP(X=train_data, Y=train_labels,
#                                              Xstar=priv_train,
#                                              C=c_value, Cstar=cstar_value)
#                     predictions_this_fold = svmplusQP_Predict(train_data, test_data, alphas, bias)
#                     scores_array[c_index,cstar_index]+= accuracy_score(test_labels, predictions_this_fold)
#
#
#
#     return scores_array
