#!/usr/bin/python
import sys
import numpy
import pdb
#import pylab as P

def getdata_AwA_one_vs_rest(PATH_data, class_id, N, test_N):	#N,test_N per class in remaining classes

    data=numpy.loadtxt(PATH_data + "all_surf.txt")
    data_star=numpy.loadtxt(PATH_data + "all_DAP.txt")
    labels = numpy.loadtxt(PATH_data + "all_labels.txt",dtype=numpy.int)
    class_id = eval(class_id)
    numclasses = 10

    X1=data[labels==class_id]
    X1_star=data_star[labels==class_id]

    if ((N+test_N)*9 > X1.shape[0]):        	#pig has 331 images, rat has 303 images
        print "Warning: total number of samples is less than required ", class_id, X1.shape[0]
        N=10; test_N = 20

    rest={}; rest_star={}
    rest_id=numpy.r_[0:class_id, class_id+1:10]	#10 classes in total, class labels 0,1,2,..,9
    for i in rest_id:
        rest[i] = data[labels==i]
        rest_star[i] = data_star[labels==i]

    idx1 = numpy.random.permutation(X1.shape[0])
    N1= N*(numclasses-1); test_N1=test_N*(numclasses-1)
    train1, test1 = idx1[:N1], idx1[N1:N1+test_N1]
    X = X1[train1]
    test_X = X1[test1]
    X_star = X1_star[train1]
    test_X_star = X1_star[test1]

    for i in rest_id:
        X2=rest[i]; X2_star = rest_star[i]
        idx2 = numpy.random.permutation(X2.shape[0])
        train2, test2 = idx2[:N], idx2[N:N+test_N]
        X = numpy.r_[X, X2[train2]];
        test_X = numpy.r_[test_X, X2[test2]]
        X_star = numpy.r_[X_star, X2_star[train2]]
        test_X_star = numpy.r_[test_X_star, X2_star[test2]]
 
    Y = numpy.r_[[1]*N1, [-1]*(N*(numclasses-1))]
    test_Y = numpy.r_[[1]*test_N1, [-1]*(test_N*(numclasses-1))]

    #L1 normalization ============================
    X = X/numpy.apply_along_axis(lambda row:numpy.linalg.norm(row,ord=1), 1, X).reshape(-1,1)
    X_star = X_star/numpy.apply_along_axis(lambda row:numpy.linalg.norm(row,ord=1), 1, X_star).reshape(-1,1)

    test_X = test_X/numpy.apply_along_axis(lambda row:numpy.linalg.norm(row,ord=1), 1, test_X).reshape(-1,1)
    test_X_star = test_X_star/numpy.apply_along_axis(lambda row:numpy.linalg.norm(row,ord=1), 1, test_X_star).reshape(-1,1)

    data=numpy.c_[X, X_star]
    test_data=numpy.c_[test_X, test_X_star]
    return numpy.asarray(data), numpy.asarray(test_data), numpy.asarray(Y), numpy.asarray(test_Y)


def getdata_arcene(PATH_data, class_id, N, test_N):	#N,test_N per class

	X1=numpy.loadtxt(PATH_data + class_id[0] + ".arcene")
	X2=numpy.loadtxt(PATH_data + class_id[1] + ".arcene")
	
	if (N+test_N > X1.shape[0]) or (N+test_N > X2.shape[0]):	
		print "Warning: total number of samples is less than required ", class_id, X1.shape[0], X2.shape[0]
		N=44; test_N = 44

        idx1 = numpy.random.permutation(X1.shape[0])     
        train1, test1 = idx1[:N], idx1[N:N+test_N]									
        idx2 = numpy.random.permutation(X2.shape[0])     
        train2, test2 = idx2[:N], idx2[N:N+test_N]							
	
	X = numpy.r_[X1[train1], X2[train2]]
	test_X = numpy.r_[X1[test1], X2[test2]]
	Y = numpy.r_[[1]*N, [-1]*N]
	test_Y = numpy.r_[[1]*test_N, [-1]*test_N]

	#L1 normalization ============================
	X = X/numpy.apply_along_axis(lambda row:numpy.linalg.norm(row,ord=1), 1, X).reshape(-1,1)
	test_X = test_X/numpy.apply_along_axis(lambda row:numpy.linalg.norm(row,ord=1), 1, test_X).reshape(-1,1)

	return numpy.asarray(X), numpy.asarray(test_X), numpy.asarray(Y), numpy.asarray(test_Y)

