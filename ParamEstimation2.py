__author__ = 'jt306'
import numpy as np
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.metrics import f1_score, pairwise
from sklearn import svm
from sklearn import grid_search
import logging
from Get_Full_Path import get_full_path


def get_gamma_from_c(c_values, features):
    euclidean_distance = pairwise.euclidean_distances(features)
    median_euclidean_distance = np.median(euclidean_distance ** 2)
    return [value / median_euclidean_distance for value in c_values]


def param_estimation(param_estimation_file, training_features, training_labels, c_values, rs, privileged,
                     privileged_training_data=None, peeking=False, testing_features=None, testing_labels=None,
                     multiplier=1, logger=None, cstar_values=None):
    training_labels=training_labels.ravel()

    dict_of_parameters = {}
    if privileged == True:
        code_with_score = np.zeros((len(c_values) ** 2))
        scores_array = np.zeros((len(c_values),len(cstar_values)))
    else:
        code_with_score = np.zeros(len(c_values))
        scores_array = np.zeros(len(c_values))

    if peeking == True:

        get_scores_for_this_fold(privileged, c_values, dict_of_parameters, training_features, training_labels,
                                 testing_features, testing_labels, privileged_training_data, code_with_score, cstar_values)

        output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)
        best_parameters = dict_of_parameters[code_with_score.argmax(axis=0)]


        return best_parameters

    else:

        print 'scores array \n', scores_array

        for train_indices, test_indices in rs:
            train_this_fold, test_this_fold = training_features[train_indices], training_features[test_indices]
            train_labels_this_fold, test_labels_this_fold = training_labels[train_indices], training_labels[test_indices]

            if privileged == True:
                privileged_train_this_fold = privileged_training_data[train_indices]
            else:
                privileged_train_this_fold = None

            scores_array = get_scores_for_this_fold(privileged, c_values, dict_of_parameters, train_this_fold, train_labels_this_fold, test_this_fold, test_labels_this_fold, privileged_train_this_fold, code_with_score, cstar_values, scores_array)
            # output_params_with_scores(dict_of_parameters, code_with_score, param_estimation_file)

        best_parameters = dict_of_parameters[code_with_score.argmax(axis=0)]

        best_indices = np.unravel_index(scores_array.argmax(), scores_array.shape)
        if privileged:
            best_parameters2 = c_values[best_indices[0]],cstar_values[best_indices[1]]
        else:
            best_parameters2 = c_values[best_indices[0]]

        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        param_estimation_file.write(np.array2string(scores_array, separator=', ').translate(None, '[]'))
        #
        # output_array = zip(c_values,scores_array)
        # print output_array
        # param_estimation_file.write('\n\n Output array:\n')
        # param_estimation_file.write(output_array)



        # output_array = np.vstack([cstar_values,np.array2string(scores_array, separator=', ').translate(None, '[]')])


        print 'dict of params, \n', dict_of_parameters
        print 'code_with_score', code_with_score


        print 'best params \n',best_parameters
        print 'best params2 \n', best_parameters2

        return best_parameters



def output_params_with_scores(dictionary, code_with_score, param_estimation_file):
    for index in range(len(code_with_score)):
        settings_with_score = str(dictionary[index]).translate(None, '[]')

        param_estimation_file.write("\n" + settings_with_score + ",")
        param_estimation_file.write(str(code_with_score[index]))
        #
        # param_estimation_file.write('\n')
        # param_estimation_file.write(','.join(map(str,code_with_score)))




def get_scores_for_this_fold(privileged,c_values,dict_of_parameters, train_data, train_labels, test_data, test_labels, priv_train, code_with_score, cstar_values, scores_array):

    j = 0


    if privileged == False:
        for c_index, c_value in enumerate(c_values):
            dict_of_parameters[j] = [c_value]
            clf = svm.SVC(C=c_value, kernel='linear')
            clf.fit(train_data, train_labels)
            code_with_score[j] += f1_score(test_labels, clf.predict(test_data))
            j += 1
            scores_array[c_index]+=f1_score(test_labels, clf.predict(test_data))


    if privileged == True:
        for c_index, c_value in enumerate(c_values):
            for cstar_index, cstar_value in enumerate(cstar_values):
                    dict_of_parameters[j] = [c_value, cstar_value]
                    alphas, bias = svmplusQP(X=train_data, Y=train_labels,
                                             Xstar=priv_train,
                                             C=c_value, Cstar=cstar_value)
                    predictions_this_fold = svmplusQP_Predict(train_data, test_data, alphas, bias)
                    code_with_score[j] += f1_score(test_labels, predictions_this_fold)
                    j += 1
                    scores_array[c_index,cstar_index]+= f1_score(test_labels, predictions_this_fold)

    return scores_array