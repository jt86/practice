__author__ = 'jt306'

from sklearn.metrics import accuracy_score, pairwise
from sklearn import svm
# from sklearn.model_selection import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
import numpy as np
import numpy
import pdb
from sklearn import linear_model, cross_validation
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
import os,sys
import time
from New import svm_problem,svm_u_problem
from Models import SVMdp, SVMu, get_accuracy_score
from sklearn.cross_validation import StratifiedKFold
from pprint import pprint

def get_best_params_dp2(setting, normal_train, labels_train, priv_train, cross_val_folder):
    n_folds=5
    c_values = setting.cvalues; gamma_values= setting.cvalues
    delta = c_values[-1]
    print(c_values)
    cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True, random_state=setting.foldnum)
    cv_scores = np.zeros((len(c_values),len(gamma_values)))
    print ('cv scores shape',cv_scores.shape)
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            for gamma_index, gamma in enumerate(gamma_values):
                print(normal_train[train].shape, priv_train[train].shape)
                print(C,gamma,delta)
                problem = svm_problem(X=normal_train[train], Xstar=priv_train[train], Y=labels_train[train].copy(), C=C, gamma=gamma, delta=delta)
                dp_classifier = SVMdp()
                # pprint(vars(problem))
                c2 = dp_classifier.train(prob=problem)
                ACC = (get_accuracy_score(c2, normal_train[test], labels_train[test]))
                cv_scores[C_index,gamma_index] += ACC
                sys.stdout.flush()
                # print (cv_scores)
    cv_scores = cv_scores/n_folds
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    print('index of best',index_of_best)
    best_C, best_gamma, best_delta  = c_values[index_of_best[0]], gamma_values[index_of_best[1]], delta
    with open(os.path.join(cross_val_folder, 'Cstar-crossvalid-{}-{}.txt'.format(setting.datasetnum, setting.topk)), 'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => best C={},best gamma={},best delta={}".format(cv_scores,best_C,best_gamma,best_delta))
    print('cross valid scores:\n',cv_scores,'=> best C',best_C, 'best gamma',best_gamma,'best delta',best_delta)
    return best_C, best_gamma, best_delta



def get_best_params_svmu(setting, normal_train, labels_train, priv_train, cross_val_folder):
    n_folds=5
    c_values = setting.cvalues; gamma_values= setting.cvalues
    delta = c_values[-1]
    print(c_values)
    cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True, random_state=setting.foldnum)
    cv_scores = np.zeros((len(c_values),len(gamma_values)))
    print ('cv scores shape',cv_scores.shape)
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            for gamma_index, gamma in enumerate(gamma_values):
                print(normal_train[train].shape, priv_train[train].shape)
                print(C,gamma,delta)
                problem = svm_problem(X=normal_train[train], Xstar=priv_train[train],XstarStar=priv_train[train], Y=labels_train[train].copy(), C=C, gamma=gamma, delta=delta)
                svmu_classifier = SVMu()
                # pprint(vars(problem))
                c2 = svmu_classifier.train(prob=problem)
                ACC = (get_accuracy_score(c2, normal_train[test], labels_train[test]))
                cv_scores[C_index,gamma_index] += ACC
                sys.stdout.flush()
                # print (cv_scores)
    cv_scores = cv_scores/n_folds
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    print('index of best',index_of_best)
    best_C, best_gamma, best_delta  = c_values[index_of_best[0]], gamma_values[index_of_best[1]], delta
    with open(os.path.join(cross_val_folder, 'Cstar-crossvalid-{}-{}.txt'.format(setting.datasetnum, setting.topk)), 'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => best C={},best gamma={},best delta={}".format(cv_scores,best_C,best_gamma,best_delta))
    print('cross valid scores:\n',cv_scores,'=> best C',best_C, 'best gamma',best_gamma,'best delta',best_delta)
    return best_C, best_gamma, best_delta


#
# def get_best_params_dp(setting, normal_train, labels_train, priv_train, cross_val_folder):
#     n_folds=5
#     c_values = setting.cvalues; gamma_values= setting.cvalues; delta_values = setting.cvalues
#     print(c_values)
#     # skf = cross_validation.StratifiedKFold(n_folds)
#     # cv = skf.split(training_data, training_labels)
#     cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True, random_state=setting.foldnum)
#     cv_scores = np.zeros((len(c_values),len(gamma_values), len(delta_values)))
#     print ('cv scores shape',cv_scores.shape)
#     for i,(train, test) in enumerate(cv):
#         for C_index, C in enumerate(c_values):
#             for gamma_index, gamma in enumerate(gamma_values):
#                 for delta_index, delta in enumerate(delta_values):
#                     problem = svm_problem(normal_train[train], priv_train[train], labels_train[train].copy(), C=C, gamma=gamma, delta=delta)
#                     dp_classifier = SVMdp()
#                     c2 = dp_classifier.train(prob=problem)
#                     ACC = (get_accuracy_score(c2, normal_train[test], labels_train[test]))
#                     cv_scores[C_index,gamma_index,delta_index] += ACC
#                     sys.stdout.flush()
#                     # print (cv_scores)
#     cv_scores = cv_scores/n_folds
#     best_positions = (np.argwhere(cv_scores.max() == cv_scores))
#     index_of_best=best_positions[0]
#     # index_of_best = best_positions[int(len(best_positions)/2)]
#     print('index of best',index_of_best)
#     best_C, best_gamma, best_delta  = c_values[index_of_best[0]], gamma_values[index_of_best[1]], delta_values[index_of_best[2]]
#     with open(os.path.join(cross_val_folder, 'Cstar-crossvalid-{}-{}.txt'.format(setting.datasetnum, setting.topk)), 'a') as cross_validation_doc:
#         cross_validation_doc.write("\n{} => best C={},best gamma={},best delta={}".format(cv_scores,best_C,best_gamma,best_delta))
#     print('cross valid scores:\n',cv_scores,'=> best C',best_C, 'best gamma',best_gamma,'best delta',best_delta)
#     return best_C, best_gamma, best_delta

def get_best_C_Cstar_omega_omegastar(s, normal_train, labels_train, priv_train, cross_val_folder2):
    omegavalues = [0.001, 1, 1000]
    cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True, random_state=s.foldnum)
    cv_scores = np.zeros((len(s.cvalues),len(s.cvalues),len(omegavalues),len(omegavalues)))	#join cross validation on X, X*
    print ('cv scores shape',cv_scores.shape)
    for i,(train, test) in enumerate(cv):
        for Cstar_index, Cstar in enumerate(s.cvalues):
            for C_index, C in enumerate(s.cvalues):
                for omegastar_index, omegastar in enumerate(omegavalues):
                    for omega_index, omega in enumerate(omegavalues):
                        # print('cstar', Cstar,'c index',C_index,'c',C)
                        duals,bias = svmplusQP(normal_train[train], labels_train[train].copy(), priv_train[train], C, Cstar, s.kernel, omega, omegastar)
                        predictions = svmplusQP_Predict(normal_train[train], normal_train[test], duals, bias).flatten()
                        ACC = np.sum(labels_train[test] == np.sign(predictions)) / (1. * len(labels_train[test]))
                        cv_scores[Cstar_index,C_index,omegastar_index,omega_index] += ACC
                        sys.stdout.flush()
    cv_scores = cv_scores/cv.n_folds

    best_C, best_Cstar, best_omega, best_omegastar = select_highest_score(s.cvalues, cv_scores, omegavalues)
    with open(os.path.join(cross_val_folder2, 'Cstar-crossvalid-{}-{}.txt'.format(s.datasetnum, s.topk)), 'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => bestC={},bestC*={}".format(cv_scores,best_C,best_Cstar))
    print('cross valid scores:\n',cv_scores,'=> best C*=',best_Cstar, 'bestC=',best_C, 'best omega*=',best_omegastar, 'bestomega=',best_omega)
    return best_C, best_Cstar, best_omega, best_omegastar

def get_best_CandCstar(s, normal_train, labels_train, priv_train, cross_val_folder2):
    if list(labels_train).count(1)<2 or list(labels_train).count(-1)<2:
        print('not enough - using default')
        return 0.001, 0.001
    if normal_train.shape[0]>=20:
        n_folds=5
    else:
        n_folds=2
    cv = StratifiedKFold(labels_train, n_folds=n_folds, shuffle=True, random_state=s.foldnum)
    cv_scores = np.zeros((len(s.cvalues),len(s.cvalues)))	#join cross validation on X, X*
    print ('cv scores shape',cv_scores.shape)
    for i,(train, test) in enumerate(cv):
        for Cstar_index, Cstar in enumerate(s.cvalues):
            for C_index, C in enumerate(s.cvalues):
                # print('cstar', Cstar,'c index',C_index,'c',C)
                duals,bias = svmplusQP(normal_train[train], labels_train[train].copy(), priv_train[train], C, Cstar, s.kernel)
                predictions = svmplusQP_Predict(normal_train[train], normal_train[test], duals, bias).flatten()
                ACC = np.sum(labels_train[test] == np.sign(predictions)) / (1. * len(labels_train[test]))
                cv_scores[Cstar_index,C_index] += ACC
                sys.stdout.flush()
    cv_scores = cv_scores/cv.n_folds

    best_C, best_Cstar = select_highest_score(s.cvalues, cv_scores)
    with open(os.path.join(cross_val_folder2, 'Cstar-crossvalid-{}-{}.txt'.format(s.datasetnum, s.topk)), 'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => bestC={},bestC*={}".format(cv_scores,best_C,best_Cstar))
    print('cross valid scores:\n',cv_scores,'=> best C*=',best_Cstar, 'bestC=',best_C)
    return best_C, best_Cstar


def get_best_params(s, all_train, labels_train, folder, method):
    # check
    if list(labels_train).count(1)<2 or list(labels_train).count(-1)<2:
        print('not enough - using default')
        return 0.001
    if all_train.shape[0]>=20:
        n_folds=5
    else:
        n_folds=2
    print('\n num folds = {}'.format(n_folds))
    cv = StratifiedKFold(labels_train, n_folds=n_folds, shuffle=True, random_state=s.foldnum)
    scores = numpy.zeros(len(s.cvalues))
    print('\n',labels_train)
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(s.cvalues):
            if method == 'rfe':
                svc = SVC(C=C, kernel="linear", random_state=s.foldnum)
                rfe = RFE(estimator=svc, n_features_to_select=s.topk, step=0.1)
                print('\n', labels_train[train])
                rfe.fit(all_train[train], labels_train[train])
                scores[C_index] += rfe.score(all_train[test], labels_train[test])
                sys.stdout.flush()
            if method == 'svm':
                print('\n', labels_train[train])
                svc = SVC(C=C, kernel="linear", random_state=s.foldnum)
                svc.fit(all_train[train], labels_train[train])
                scores[C_index] += svc.score(all_train[test], labels_train[test])
                sys.stdout.flush()
    scores = scores/cv.n_folds

    C = select_highest_score(s.cvalues, scores)
    with open(os.path.join(folder, 'crossval-{}.txt'.format(s.skfseed)),'a') as cross_val_doc:
        cross_val_doc.write("\n {}  {} \n {}".format(s.foldnum,C,scores))
    return C

def select_highest_score(cvalues, scores, omegavalues=None):
    best_positions = (np.argwhere(scores.max() == scores))
    # index_of_best  =best_positions[(len(best_positions)//2)] # = best_positions[0]
    index_of_best = best_positions[0]
    print('index of best',index_of_best)
    print('length of scores',len(scores.shape))
    if len(scores.shape) == 1:
        return cvalues[index_of_best]
    if len(scores.shape) == 2:
        return cvalues[index_of_best[1]], cvalues[index_of_best[0]]
    if len(scores.shape) == 4:
        return cvalues[index_of_best[1]], cvalues[index_of_best[0]],omegavalues[index_of_best[3]], omegavalues[index_of_best[2]]
