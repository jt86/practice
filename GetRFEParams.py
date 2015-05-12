__author__ = 'jt306'
import sys
import numpy as np
import numpy
import pdb
from sklearn import cross_validation, linear_model
from scipy.optimize import *
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from Get_Awa_Data import get_awa_data
import time
from Get_Full_Path import get_full_path
import os
import argparse

def get_rfe_params(dataset, top_k_percent, k):
    numpy.random.seed(k)
    all_training, all_testing, training_labels, testing_labels = get_awa_data("", dataset[-1])
    print all_training[0]

    reg_array= [1.0, 10., 100., 1000., 10000]

    output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/FixedCandCStar4/'),dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print top_k_percent
    top_k_features = all_training.shape[1]*top_k_percent/100
    X = all_training.copy(); Y = training_labels.copy()
    cv_scores = numpy.zeros(len(reg_array))
    cv = cross_validation.StratifiedKFold(Y, 3)

    for i,(train, test) in enumerate(cv):
        t0=time.clock()
        print 'i=',i
        for j, reg_const in enumerate(reg_array):
            print 'j=',j
            svc = SVC(C=reg_const, kernel="linear", random_state=1)
            rfe = RFE(estimator=svc, n_features_to_select=top_k_features, step=1)
            rfe.fit(X[train], Y[train])
            cv_scores[j] = cv_scores[j] + rfe.score(X[test], Y[test])
            print 'time elapsed for fold',i,j,'=',time.clock()-t0
            print cv_scores
    cv_scores = cv_scores/5.
    reg_best = reg_array[numpy.argmax(cv_scores)]

    with open (os.path.join(output_directory,str(top_k_percent)+'.txt'),'a') as best_rfe_param_file:
        best_rfe_param_file.write(str(reg_best))


    print "svm+rfe Regularization:", reg_best
    return reg_best





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataset', type=str, required=True, help='name of input data')
    parser.add_argument('--k', type=int, required = True, help='fold number - used to seed')
    parser.add_argument('--top-k-percent', type=int, required=True, help='percentage of features to take')

    args = parser.parse_args()
    print 'input is', args.dataset
    print ' all args',args

    get_rfe_params(args.dataset, args.top_k_percent, args.k)
