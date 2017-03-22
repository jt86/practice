__author__ = 'jt306'

from sklearn.metrics import accuracy_score, pairwise
from sklearn import svm
from sklearn.model_selection import StratifiedKFold,KFold, ShuffleSplit, StratifiedShuffleSplit
import numpy as np
import numpy
import pdb
from sklearn import model_selection, linear_model
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
import os,sys
import time
from New import svm_problem,svm_u_problem
from Models import SVMdp, SVMu, get_accuracy_score


def get_best_params_dp(training_data, training_labels, privileged_data, c_values, gamma_values, delta_values, cross_validation_folder, datasetnum, topk):
    n_folds=5
    skf = model_selection.StratifiedKFold(n_folds)
    cv = skf.split(training_data, training_labels)
    cv_scores = np.zeros((len(c_values),len(gamma_values), len(delta_values)))
    print ('cv scores shape',cv_scores.shape)

    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            for gamma_index, gamma in enumerate(gamma_values):
                for delta_index, delta in enumerate(delta_values):

                    problem = svm_problem(training_data[train], privileged_data[train], training_labels[train].copy(), C=C, gamma=gamma, delta=delta)
                    dp_classifier = SVMdp()
                    c2 = dp_classifier.train(prob=problem)
                    ACC = (get_accuracy_score(c2, training_data[test], training_labels[test]))

                    cv_scores[C_index,gamma_index,delta_index] += ACC
                    sys.stdout.flush()
                    # print (cv_scores)
    cv_scores = cv_scores/n_folds
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]

    print('index of best',index_of_best)
    best_C, best_delta, best_gamma  = c_values[index_of_best[0]], gamma_values[index_of_best[1]], delta_values[index_of_best[2]]
    with open(os.path.join(cross_validation_folder,'Cstar-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => best C={},best gamma={},best delta={}".format(cv_scores,best_C,best_gamma,best_delta))
    print('cross valid scores:\n',cv_scores,'=> best C',best_C, 'best delta',best_delta, 'best gamma',best_gamma)
    return best_C, best_gamma, best_delta


def get_best_CandCstar(training_data,training_labels, privileged_data, c_values, Cstar_values,cross_validation_folder,datasetnum,topk):
    n_folds=5
    skf = model_selection.StratifiedKFold(n_folds)
    cv = skf.split(training_data, training_labels)
    cv_scores = np.zeros((len(Cstar_values),len(c_values)))	#join cross validation on X, X*
    print ('cv scores shape',cv_scores.shape)

    for i,(train, test) in enumerate(cv):
        for Cstar_index, Cstar in enumerate(Cstar_values):
            for C_index, C in enumerate(c_values):
                # print('cstar', Cstar,'c index',C_index,'c',C)
                duals,bias = svmplusQP(training_data[train],training_labels[train].copy(),privileged_data[train],C,Cstar)
                predictions = svmplusQP_Predict(training_data[train],training_data[test],duals,bias).flatten()
                ACC = np.sum(training_labels[test]==np.sign(predictions))/(1.*len(training_labels[test]))
                cv_scores[Cstar_index,C_index] += ACC
                sys.stdout.flush()
                # print (cv_scores)
    cv_scores = cv_scores/n_folds
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]

    print('index of best',index_of_best)
    best_Cstar, best_C = Cstar_values[index_of_best[0]],c_values[index_of_best[1]]
    with open(os.path.join(cross_validation_folder,'Cstar-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} => bestC={},bestC*={}".format(cv_scores,best_C,best_Cstar))
    print('c* values:',Cstar_values)
    print('c values:',c_values)
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

def get_best_C(training_data,training_labels, c_values, cross_validation_folder,datasetnum,topk):
    skf = model_selection.StratifiedKFold(5)
    cv = skf.split(training_data,training_labels)
    cv_scores = np.zeros(len(c_values))
    for i,(train, test) in enumerate(cv):
        for C_index, C in enumerate(c_values):
            # print('c index',C_index,'c',C)
            svc = SVC(C=C, kernel="linear", random_state=1)
            svc.fit(training_data[train], training_labels[train])
            cv_scores[C_index] += svc.score(training_data[test], training_labels[test])
            sys.stdout.flush()
    cv_scores = cv_scores/5.
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    best_C = c_values[index_of_best][0]
    with open(os.path.join(cross_validation_folder,'C_fullset-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
        cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    print('cross valid scores (all features):',cv_scores,'=> best C=',best_C)
    return best_C



def get_best_RFE_C(training_data,training_labels, c_values, top, stepsize,datasetnum,topk):
    starttime = time.clock()
    # print ('time', starttime)
    skf = model_selection.StratifiedKFold(5)
    cv = skf.split(training_data, training_labels)
    cv_scores = numpy.zeros(len(c_values))

    for i,(train, test) in enumerate(cv):
        # print('iter',i)
        for C_index, C in enumerate(c_values):
            # print('c index',C_index,'c',C)
            svc = SVC(C=C, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=top, step=stepsize)
            rfe.fit(training_data[train], training_labels[train])
            cv_scores[C_index] += rfe.score(training_data[test], training_labels[test])
            sys.stdout.flush()
        # print 'fold',i,cv_scores

    cv_scores = cv_scores/5.
    best_positions = (np.argwhere(cv_scores.max() == cv_scores))
    index_of_best=best_positions[0]
    # index_of_best = best_positions[int(len(best_positions)/2)]
    best_C = c_values[index_of_best]
    # with open(os.path.join(cross_validation_folder,'C_fullset-crossvalid-{}-{}.txt'.format(datasetnum,topk)),'a') as cross_validation_doc:
    #     cross_validation_doc.write("\n{} {}".format(cv_scores,best_C))
    print('cross valid scores (rfe):',cv_scores,'=> best C=',best_C)
    print ('-------\ntime',time.clock()-starttime,'\n-------')
    # sys.exit()
    return best_C

