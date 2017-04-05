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

# def get_best_params_dp2(setting, training_data, training_labels, privileged_data, c_values, cross_val_folder):
#     gamma_values = c_values
#     delta = c_values[-1]
#     n_folds=5
#     # skf = cross_validation.StratifiedKFold(n_folds)
#     # cv = skf.split(training_data, training_labels)
#     cv = StratifiedKFold(training_labels, n_folds=5, shuffle=True)
#     cv_scores = np.zeros((len(c_values),len(gamma_values)))
#     print ('cv scores shape',cv_scores.shape)
#     for i,(train, test) in enumerate(cv):
#         for C_index, C in enumerate(c_values):
#             for gamma_index, gamma in enumerate(gamma_values):
#                 problem = svm_problem(training_data[train], privileged_data[train], training_labels[train].copy(), C=C, gamma=gamma, delta=delta)
#                 dp_classifier = SVMdp()
#                 c2 = dp_classifier.train(prob=problem)
#                 ACC = (get_accuracy_score(c2, training_data[test], training_labels[test]))
#                 cv_scores[C_index,gamma_index] += ACC
#                 sys.stdout.flush()
#                 # print (cv_scores)
#     cv_scores = cv_scores/n_folds
#     best_positions = (np.argwhere(cv_scores.max() == cv_scores))
#     index_of_best=best_positions[0]
#     # index_of_best = best_positions[int(len(best_positions)/2)]
#
#     print('index of best',index_of_best)
#     best_C, best_gamma  = c_values[index_of_best[0]], gamma_values[index_of_best[1]]
#
#     with open(os.path.join(cross_val_folder, 'Cstar-crossvalid-{}-{}.txt'.format(setting.datasetnum, setting.topk)), 'a') as cross_validation_doc:
#         cross_validation_doc.write("\n{} => best C={},best gamma={},best delta={}".format(cv_scores,best_C,best_gamma,delta))
#     print('cross valid scores:\n',cv_scores,'=> best C',best_C,  'best gamma',best_gamma, 'fixed delta',delta)
#     return best_C, best_gamma, delta

def get_best_params_dp(setting, normal_train, labels_train, priv_train, cross_val_folder):
    n_folds=5
    c_values = setting.cvalues; gamma_values= setting.cvalues; delta_values = setting.cvalues
    print(c_values)
    # skf = cross_validation.StratifiedKFold(n_folds)
    # cv = skf.split(training_data, training_labels)
    cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True)
    cv_scores = np.zeros((len(c_values),len(gamma_values), len(delta_values)))
    print ('cv scores shape',cv_scores.shape)
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            for gamma_index, gamma in enumerate(gamma_values):
                for delta_index, delta in enumerate(delta_values):
                    problem = svm_problem(normal_train[train], priv_train[train], labels_train[train].copy(), C=C, gamma=gamma, delta=delta)
                    dp_classifier = SVMdp()
                    c2 = dp_classifier.train(prob=problem)
                    ACC = (get_accuracy_score(c2, normal_train[test], labels_train[test]))
                    cv_scores[C_index,gamma_index,delta_index] += ACC
                    sys.stdout.flush()
                    # print (cv_scores)
    cv_scores = cv_scores/n_folds
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    print('index of best',index_of_best)
    best_C, best_gamma, best_delta  = c_values[index_of_best[0]], gamma_values[index_of_best[1]], delta_values[index_of_best[2]]
    with open(os.path.join(cross_val_folder, 'Cstar-crossvalid-{}-{}.txt'.format(setting.datasetnum, setting.topk)), 'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => best C={},best gamma={},best delta={}".format(cv_scores,best_C,best_gamma,best_delta))
    print('cross valid scores:\n',cv_scores,'=> best C',best_C, 'best gamma',best_gamma,'best delta',best_delta)
    return best_C, best_gamma, best_delta



def get_best_CandCstar(s, normal_train, labels_train, priv_train, cross_val_folder2):
    n_folds=5
    # skf = cross_validation.StratifiedKFold(n_folds)
    # cv = skf.split(training_data, training_labels)
    cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True)
    cv_scores = np.zeros((len(s.cvalues),len(s.cvalues)))	#join cross validation on X, X*
    print ('cv scores shape',cv_scores.shape)
    for i,(train, test) in enumerate(cv):
        for Cstar_index, Cstar in enumerate(s.cvalues):
            for C_index, C in enumerate(s.cvalues):
                # print('cstar', Cstar,'c index',C_index,'c',C)
                duals,bias = svmplusQP(normal_train[train], labels_train[train].copy(), priv_train[train], C, Cstar)
                predictions = svmplusQP_Predict(normal_train[train], normal_train[test], duals, bias).flatten()
                ACC = np.sum(labels_train[test] == np.sign(predictions)) / (1. * len(labels_train[test]))
                cv_scores[Cstar_index,C_index] += ACC
                sys.stdout.flush()
                # print (cv_scores)
    cv_scores = cv_scores/n_folds
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]

    print('index of best',index_of_best)
    best_Cstar, best_C = s.cvalues[index_of_best[0]],s.cvalues[index_of_best[1]]
    with open(os.path.join(cross_val_folder2, 'Cstar-crossvalid-{}-{}.txt'.format(s.datasetnum, s.topk)), 'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => bestC={},bestC*={}".format(cv_scores,best_C,best_Cstar))
    print('c* values:',s.cvalues)
    print('c values:',s.cvalues)
    print('cross valid scores:\n',cv_scores,'=> best C*=',best_Cstar, 'bestC=',best_C)
    return best_C, best_Cstar


# def get_best_Cstar(training_data,training_labels, privileged_data, C, Cstar_values,cross_validation_folder,datasetnum,topk):
#     n_folds=5
#     cv = cross_validation.StratifiedKFold(training_labels, n_folds)
#     cv_scores = np.zeros(len(Cstar_values))	#join cross validation on X, X*
#     for i,(train, test) in enumerate(cv):
#         for Cstar_index, Cstar in enumerate(Cstar_values):
#             duals,bias = svmplusQP(training_data[train],training_labels[train].copy(),privileged_data[train],C,Cstar)
#             predictions = svmplusQP_Predict(training_data[train],training_data[test],duals,bias).flatten()
#             ACC = np.sum(training_labels[test]==np.sign(predictions))/(1.*len(training_labels[test]))
#             cv_scores[Cstar_index] += ACC
#             sys.stdout.flush()
#     cv_scores = cv_scores/n_folds
#     best_positions = (np.argwhere(cv_scores.max() == cv_scores))
#     index_of_best=best_positions[0]
#     # index_of_best = best_positions[int(len(best_positions)/2)]
#     best_Cstar = Cstar_values[index_of_best]
#     with open(os.path.join(cross_validation_folder,'Cstar-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
#         cross_validation_doc.write("\n{} {}".format(cv_scores,best_Cstar))
#     print('cross valid scores:',cv_scores,'=> best C*=',best_Cstar)
#     return best_Cstar

def get_best_C(s, all_train, labels_train, cross_val_folder):
    # skf = cross_validation.StratifiedKFold(5)
    # cv = skf.split(training_data,training_labels)
    cv =StratifiedKFold(labels_train, n_folds=5, shuffle=True)
    cv_scores = np.zeros(len(s.cvalues))
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(s.cvalues):
            # print('c index',C_index,'c',C)
            svc = SVC(C=C, kernel="linear", random_state=1)
            svc.fit(all_train[train], labels_train[train])
            cv_scores[C_index] += svc.score(all_train[test], labels_train[test])
            sys.stdout.flush()
    cv_scores = cv_scores/5.
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    best_C = s.cvalues[index_of_best][0]
    with open(os.path.join(cross_val_folder,'C_fullset-crossvalid-{}-{}.txt'.format(s.datasetnum,s.topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    print('cross valid scores (all features):',cv_scores,'=> best C=',best_C)
    return best_C

    # best_rfe_param = get_best_RFE_C(all_training, labels_train, s.cvalues, s.topk, stepsize)


def get_best_RFE_C(setting, all_train, labels_train):

    starttime = time.clock()
    # print ('time', starttime)
    # skf = cross_validation.StratifiedKFold(5)
    # cv = skf.split(training_data, training_labels)
    cv = StratifiedKFold(labels_train, n_folds=5, shuffle=True)
    cv_scores = numpy.zeros(len(setting.cvalues))

    for i,(train, test) in enumerate(cv):
        # print('iter',i)
        for C_index, C in enumerate(setting.cvalues):
            # print('c index',C_index,'c',C)
            svc = SVC(C=C, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=setting.topk, step=setting.stepsize)
            rfe.fit(all_train[train], labels_train[train])
            cv_scores[C_index] += rfe.score(all_train[test], labels_train[test])
            sys.stdout.flush()
        # print 'fold',i,cv_scores

    cv_scores = cv_scores/5.
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    best_C = setting.cvalues[index_of_best]
    # with open(os.path.join(cross_validation_folder,'C_fullset-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
    #     cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    print('cross valid scores (rfe):',cv_scores,'=> best C=',best_C)
    print ('-------\ntime',time.clock()-starttime,'\n-------')
    # sys.exit()
    return best_C

