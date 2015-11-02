__author__ = 'jt306'
import numpy as np
import numpy
from sklearn import cross_validation, linear_model
from SVMplus4 import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
import os

def get_best_Cstar(training_data,training_labels, privileged_data, C, Cstar_values,cross_validation_folder,datasetnum,topk):
    n_folds=5
    cv = cross_validation.StratifiedKFold(training_labels, n_folds)
    print('shape',training_data.shape[0])
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
    return best_C

def get_best_RFE_C(training_data,training_labels, c_values, top, stepsize,cross_validation_folder,datasetnum,topk):
    cv = cross_validation.StratifiedKFold(training_labels, 5)
    cv_scores = numpy.zeros(len(c_values))
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            svc = SVC(C=C, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=top, step=stepsize)
            rfe.fit(training_data[train], training_labels[train])
            cv_scores[C_index] += rfe.score(training_data[test], training_labels[test])
    cv_scores = cv_scores/5.
    index_of_best = np.argwhere(cv_scores.max() == cv_scores)[0]
    best_C = c_values[index_of_best]
    with open(os.path.join(cross_validation_folder,'C_RFE-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    return best_C


