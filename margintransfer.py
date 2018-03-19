#!/usr/bin/python
#
#V.Sharmanska, N.Quadrianto, C.H. Lampert: 
#"Learning to Transfer Privileged Information", arXiv:1410.0389v1, 2014.
#
#Algorithm: Margin Transfer from X* to X.
#INPUT: data X, privileged data X*, labels Y, tolerance tol.
#STEP1 Train SVM on privileged data (X*,Y): return f*(x*). 
#STEP2 Compute (per-sample margin) rho_i = max {y_i f*(x_i*), tol}.
#STEP3 Train SVM on data (X, Y) using rho_i instead of unit margin.
#      This is equivalent to solving the weighted SVM (4a)-(4b) 
#      with the transformed data samples x_i = x_i/rho_i.
#RETURN: f(x).

import sys
import numpy
import liblinearutil
#from margintransfer_cv import do_CV_5x5fold 	#uncomment this line to include cross validation

def transform(X, Y, fstar, tol=1e-1):					
	rho=numpy.zeros(X.shape[0])
	Xi=X.copy()
	for i in range(X.shape[0]):
		rho[i] = max(Y[i]*numpy.float32(fstar[i]), tol)		
		Xi[i] = Xi[i]/rho[i]	
	return Xi, rho

def main():
	
	print ("***Loading the data***")
	X = numpy.loadtxt("data/X.txt")
	Xstar = numpy.loadtxt("data/Xstar.txt")
	testX = numpy.loadtxt("data/testX.txt")
	Y = numpy.loadtxt("data/Y.txt")
	testY = numpy.loadtxt("data/testY.txt")

	print ("***Margin Transfer method***")
	C=100.; Cstar=0.01 
	#to run cross validation on C and Cstar, uncomment line 40:
	#(C, Cstar) = do_CV_5x5fold(X,Y, Xstar, [1., 10., 100.], [0.01, 0.1, 1.])

	print ("STEP1: Training SVM on privileged data (X*,Y)")
	m_star = liblinearutil.train(Y.tolist(), Xstar.tolist() , '-s 3 -c %f -e 0.001 -q'%Cstar) # removed first arg: []

	print ("STEP2: Computing per-sample margin and transforming the data")
	fstar = liblinearutil.predict(Y.tolist(), Xstar.tolist(), m_star, '-q')[-1]
	fstar = numpy.array(fstar).reshape(-1,)
	Xtrans, W = transform(X, Y, fstar)

	print ("STEP3: Training weighted SVM on data (Xtrans,Y)")
	m = liblinearutil.train(W.tolist(), Y.tolist(), Xtrans.tolist(), '-s 3 -c %f -e 0.001 -q'%C)

	#Compute the accuracy
	p_label, p_acc, p_val = liblinearutil.predict(testY.tolist(), testX.tolist(), m, '-q')
	print ('Margin Transfer Accuracy: %f'%(p_acc[0]))

	#As a reference, we compute the performance of SVM on (X,Y):
	m = liblinearutil.train([], Y.tolist(), X.tolist(), '-s 3 -c %f -e 0.001 -q'%C)
	p_label, p_acc, p_val = liblinearutil.predict(testY.tolist(), testX.tolist(), m, '-q')
	print ('As a reference, SVM Accuracy: %f'%(p_acc[0]))
	
if __name__ == '__main__':
	main()
	
