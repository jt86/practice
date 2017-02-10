#!/usr/bin/python
import sys
import numpy
import liblinearutil
from sklearn import cross_validation

def transform(X, Y, fstar, tol=1e-1):					
	rho=numpy.zeros(X.shape[0])
	Xi=X.copy()
	for i in xrange(X.shape[0]):
		rho[i] = max(Y[i]*numpy.float32(fstar[i]), tol)		
		Xi[i] = Xi[i]/rho[i]	
	return Xi, rho

def do_CV_5x5fold(Xorig,Yorig, X_starorig, reg_array, reg_array_star):

	print "***5x5 fold cross validation***"
	cv_scores = numpy.zeros( (len(reg_array), len(reg_array_star) ) )
	#5 random permutations	
	for rep in xrange(5):	
		rep_idx=numpy.random.permutation(len(Yorig))     
		Y = Yorig[rep_idx].copy()
		X = Xorig[rep_idx].copy()
		X_star = X_starorig[rep_idx].copy()
		#5 fold
		cv = cross_validation.StratifiedKFold(Y, 5)	
		for i,(train, test) in enumerate(cv):		
			for j, C in enumerate(reg_array):
				for l, Cstar in enumerate(reg_array_star):
					m_star = liblinearutil.train([], Y[train].tolist(), X_star[train].tolist() , '-s 3 -c %f -e 0.001 -q'%Cstar)

					fstar = liblinearutil.predict(Y.tolist(), X_star.tolist(), m_star, '-q')[-1]
					fstar = numpy.array(fstar).reshape(-1,)
					X_trans, W = transform(X[train], Y[train], fstar[train])

					m = liblinearutil.train(W.tolist(), Y[train].tolist(), X_trans.tolist(), '-s 3 -c %f -e 0.001 -q'%C)
					p_label, p_acc, p_val = liblinearutil.predict(Y[test].tolist(), X[test].tolist(), m, '-q')
					cv_scores[j,l] = cv_scores[j,l] + p_acc[0]
					del m_star
					del m		
	cv_scores = cv_scores/25.	#5x5 cross validation	
	idx = numpy.argwhere(cv_scores.max() == cv_scores)[0]		
	reg_best = reg_array[idx[0]]
	reg_best_star = reg_array_star[idx[1]]
	print "Best C, Cstar:", reg_best, reg_best_star
	return reg_best, reg_best_star

