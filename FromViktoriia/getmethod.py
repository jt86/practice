#!/usr/bin/python
import sys
import numpy as np
import numpy
import pdb
from sklearn import cross_validation, linear_model
from scipy.optimize import *
from SVMplus import svmplusQP, svmplusQP_Predict
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC

def do_CV_svmrfe_5fold(Xorig,Yorig, reg_array, dataset, PATH_CV_results, method, class_id, k, top):

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
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, numpy.asarray([[reg_best]]), fmt='%.5f')		
	return reg_best


def do_CV_svml1_5fold(Xorig,Yorig, reg_array, dataset, PATH_CV_results, method, class_id, k):

	X = Xorig.copy(); Y = Yorig.copy()	
	cv_scores = numpy.zeros(len(reg_array))	
	cv = cross_validation.StratifiedKFold(Y, 5)
	for i,(train, test) in enumerate(cv):		
		for j, reg_const in enumerate(reg_array):
			svc = LinearSVC(C=reg_const, penalty="l1", dual=False, random_state=1)
			svc.fit(X[train], Y[train])
			cv_scores[j] = cv_scores[j] + svc.score(X[test], Y[test])		
	cv_scores = cv_scores/5.	
	reg_best = reg_array[numpy.argmax(cv_scores)]
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, numpy.asarray([[reg_best]]), fmt='%.5f')	
	
	return reg_best

def do_CV_logreg_5fold(Xorig,Yorig, reg_array, dataset, PATH_CV_results, method, class_id, k):

	X = Xorig.copy(); Y = Yorig.copy()	
	cv_scores = numpy.zeros(len(reg_array))	
	cv = cross_validation.StratifiedKFold(Y, 5)
	for i,(train, test) in enumerate(cv):		
		for j, reg_const in enumerate(reg_array):
			svc = linear_model.LogisticRegression(C=reg_const, penalty="l1", dual=False, random_state=1)
			svc.fit(X[train], Y[train])
			cv_scores[j] = cv_scores[j] + svc.score(X[test], Y[test])		
	cv_scores = cv_scores/5.	
	reg_best = reg_array[numpy.argmax(cv_scores)]
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, numpy.asarray([[reg_best]]), fmt='%.5f')	
	
	return reg_best

def do_CV_logreg_5x5fold(Xorig,Yorig, reg_array, dataset, PATH_CV_results, method, class_id, k):

	X = Xorig.copy(); Y = Yorig.copy()	
	cv_scores = numpy.zeros(len(reg_array))	
	for rep in range(5):
		rep_idx=numpy.random.permutation(len(Y))  
		Y = Y[rep_idx].copy()
		X = X[rep_idx].copy()
		cv = cross_validation.StratifiedKFold(Y, 5)
		for i,(train, test) in enumerate(cv):		
			for j, reg_const in enumerate(reg_array):
				svc = linear_model.LogisticRegression(C=reg_const, penalty="l1", dual=False, random_state=1)
				svc.fit(X[train], Y[train])
				cv_scores[j] = cv_scores[j] + svc.score(Y[test], X[test])
	cv_scores = cv_scores/25.	
	reg_best = reg_array[numpy.argmax(cv_scores)]
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, numpy.asarray([[reg_best]]), fmt='%.5f')		
	return reg_best


def do_CV_svm_plus_5fold(X,Y, X_star, reg_array, reg_array_star, dataset, PATH_CV_results, method, class_id, k):

	cv = cross_validation.StratifiedKFold(Y, 5)
	cv_scores = np.zeros( (len(reg_array), len(reg_array_star) ) )	#join cross validation on X, X*
	
	for i,(train, test) in enumerate(cv):		
		for j, reg_const in enumerate(reg_array):
			for l, reg_const_star in enumerate(reg_array_star):

				duals,bias = svmplusQP(X[train],Y[train].copy(),X_star[train],reg_const,reg_const_star)
				testXranked = svmplusQP_Predict(X[train],X[test],duals,bias).flatten()
				ACC = np.sum(Y[test]==np.sign(testXranked))/(1.*len(Y[test])) 
				cv_scores[j,l] = cv_scores[j,l] + ACC
		
	cv_scores = cv_scores/5.	
	idx = np.argwhere(cv_scores.max() == cv_scores)[0]		
	reg_best = reg_array[idx[0]]
	reg_best_star = reg_array_star[idx[1]]
	np.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	np.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, np.asarray([[reg_best]]), fmt='%.5f')
	np.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best_star.txt"%k, np.asarray([[reg_best_star]]), fmt='%.5f')	

	return reg_best, reg_best_star


def do_CV_svm_plus_5x5fold(X,Y, X_star, reg_array, reg_array_star, dataset, PATH_CV_results, method, class_id, k):

        cv_scores = np.zeros( (len(reg_array), len(reg_array_star) ) )  #join cross validation on X, X*

	for rep in xrange(5):
		rep_idx=np.random.permutation(len(Y))
		Y = Y[rep_idx].copy()
		X = X[rep_idx].copy()
		X_star = X_star[rep_idx].copy()
		cv = cross_validation.StratifiedKFold(Y, 5)
        	for i,(train, test) in enumerate(cv):
                	for j, reg_const in enumerate(reg_array):
                        	for l, reg_const_star in enumerate(reg_array_star):

                                	duals,bias = svmplusQP(X[train],Y[train].copy(),X_star[train],reg_const,reg_const_star)
                                	testXranked = svmplusQP_Predict(X[train],X[test],duals,bias).flatten()
                                	ACC = np.sum(Y[test]==np.sign(testXranked))/(1.*len(Y[test]))
                                	cv_scores[j,l] = cv_scores[j,l] + ACC

        cv_scores = cv_scores/25.
        idx = np.argwhere(cv_scores.max() == cv_scores)[0]
        reg_best = reg_array[idx[0]]
        reg_best_star = reg_array_star[idx[1]]
        np.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
        np.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, np.asarray([[reg_best]]), fmt='%.5f')
        np.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best_star.txt"%k, np.asarray([[reg_best_star]]), fmt='%.5f')

        return reg_best, reg_best_star


def do_CV_svm_5fold(Xorig,Yorig, reg_array, dataset, PATH_CV_results, method, class_id, k):

	X = Xorig.copy(); Y = Yorig.copy()	
	cv_scores = numpy.zeros(len(reg_array))	
	cv = cross_validation.StratifiedKFold(Y, 5)
	for i,(train, test) in enumerate(cv):		
		for j, reg_const in enumerate(reg_array):
			svc = SVC(C=reg_const, kernel="linear", random_state=1)
			svc.fit(X[train], Y[train])
			cv_scores[j] = cv_scores[j] + svc.score(X[test], Y[test])	
	cv_scores = cv_scores/5.	
	reg_best = reg_array[numpy.argmax(cv_scores)]
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, numpy.asarray([[reg_best]]), fmt='%.5f')		
	return reg_best

def do_CV_svm_5x5fold(Xorig,Yorig, reg_array, dataset, PATH_CV_results, method, class_id, k):

	X = Xorig.copy(); Y = Yorig.copy()	
	cv_scores = numpy.zeros(len(reg_array))	
	for rep in xrange(5):
		rep_idx=numpy.random.permutation(len(Y))  
		Y = Y[rep_idx].copy()
		X = X[rep_idx].copy()
		cv = cross_validation.StratifiedKFold(Y, 5)
		for i,(train, test) in enumerate(cv):		
			for j, reg_const in enumerate(reg_array):
				svc = SVC(C=reg_const, kernel="linear", random_state=1)
				svc.fit(X[train], Y[train])
				cv_scores[j] = cv_scores[j] + svc.score(X[test], Y[test])
	cv_scores = cv_scores/25.	
	reg_best = reg_array[numpy.argmax(cv_scores)]
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_cv.txt"%k, cv_scores, fmt='%.5f')
	numpy.savetxt(PATH_CV_results + dataset + "_" + method + "_" +class_id + "class_"+ "%ddata_best.txt"%k, numpy.asarray([[reg_best]]), fmt='%.5f')		
	return reg_best

