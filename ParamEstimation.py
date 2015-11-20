__author__ = 'jt306'

from sklearn.metrics import accuracy_score, pairwise
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
import numpy as np
import numpy
import pdb
from sklearn import cross_validation, linear_model
from SVMplus4 import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
import os,sys


def get_best_CandCstar(training_data,training_labels, privileged_data, c_values, Cstar_values,cross_validation_folder,datasetnum,topk):
    n_folds=5
    cv = cross_validation.StratifiedKFold(training_labels, n_folds)
    cv_scores = np.zeros((len(Cstar_values),len(c_values)))	#join cross validation on X, X*
    print ('cv scores shape',cv_scores.shape)

    for i,(train, test) in enumerate(cv):
        for Cstar_index, Cstar in enumerate(Cstar_values):
            for C_index, C in enumerate(c_values):
                duals,bias = svmplusQP(training_data[train],training_labels[train].copy(),privileged_data[train],C,Cstar)
                predictions = svmplusQP_Predict(training_data[train],training_data[test],duals,bias).flatten()
                ACC = np.sum(training_labels[test]==np.sign(predictions))/(1.*len(training_labels[test]))
                cv_scores[Cstar_index,C_index] += ACC
                # print (cv_scores)
    cv_scores = cv_scores/n_folds
    index_of_best = np.argwhere(cv_scores.max() == cv_scores)[0]
    best_Cstar, best_C = Cstar_values[index_of_best[0]],c_values[index_of_best[1]]
    with open(os.path.join(cross_validation_folder,'Cstar-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_Cstar))
    print('c* values:',Cstar_values)
    print('c values:',c_values)
    print('cross valid scores:\n',cv_scores,'=> best C*=',best_Cstar, 'bestC=',best_C)
    return best_C, best_Cstar


def get_best_Cstar(training_data,training_labels, privileged_data, C, Cstar_values,cross_validation_folder,datasetnum,topk):
    n_folds=5
    cv = cross_validation.StratifiedKFold(training_labels, n_folds)
    cv_scores = np.zeros(len(Cstar_values))	#join cross validation on X, X*
    for i,(train, test) in enumerate(cv):
        for Cstar_index, Cstar in enumerate(Cstar_values):
            duals,bias = svmplusQP(training_data[train],training_labels[train].copy(),privileged_data[train],C,Cstar)
            predictions = svmplusQP_Predict(training_data[train],training_data[test],duals,bias).flatten()
            ACC = np.sum(training_labels[test]==np.sign(predictions))/(1.*len(training_labels[test]))
            cv_scores[Cstar_index] += ACC
    cv_scores = cv_scores/n_folds
    index_of_best = np.argwhere(cv_scores.max() == cv_scores)[0]
    best_Cstar = Cstar_values[index_of_best]
    with open(os.path.join(cross_validation_folder,'Cstar-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_Cstar))
    print('cross valid scores:',cv_scores,'=> best C*=',best_Cstar)
    return best_Cstar

def get_best_C(training_data,training_labels, c_values, cross_validation_folder,datasetnum,topk):
    cv = cross_validation.StratifiedKFold(training_labels, 5)
    cv_scores = np.zeros(len(c_values))
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            svc = SVC(C=C, kernel="linear", random_state=1)
            svc.fit(training_data[train], training_labels[train])
            cv_scores[C_index] += svc.score(training_data[test], training_labels[test])
    cv_scores = cv_scores/5.
    index_of_best = np.argwhere(cv_scores.max() == cv_scores)[0]
    best_C = c_values[index_of_best][0]
    with open(os.path.join(cross_validation_folder,'C_fullset-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    print('cross valid scores (all featurers):',cv_scores,'=> best C=',best_C)
    return best_C



def get_best_RFE_C(training_data,training_labels, c_values, top, stepsize,cross_validation_folder,datasetnum,topk):
    cv = cross_validation.StratifiedKFold(training_labels, 5)
    cv_scores = numpy.zeros(len(c_values))

    for i,(train, test) in enumerate(cv):
        # print('iter',i)
        for C_index, C in enumerate(c_values):
            # print('c index',C_index)
            svc = SVC(C=C, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=top, step=stepsize)
            rfe.fit(training_data[train], training_labels[train])
            cv_scores[C_index] += rfe.score(training_data[test], training_labels[test])
        # print 'fold',i,cv_scores

    cv_scores = cv_scores/5.
    index_of_best = np.argwhere(cv_scores.max() == cv_scores)[0]
    best_C = c_values[index_of_best]
    with open(os.path.join(cross_validation_folder,'C_fullset-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    print('cross valid scores (rfe):',cv_scores,'=> best C=',best_C)
    return best_C


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


