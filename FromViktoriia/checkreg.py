#!/usr/bin/python
import sys
import os
import pdb
import numpy
import scipy.stats
#import sklearn

PATH = "AwA/"
PATH_AP = PATH + "Results/"
PATH_CV_results = PATH + "CV/"
dataset="AwA"
all_classes = ['0','1', '2', '3', '4', '5', '6', '7', '8','9']
all_data = [1,2,3,4,5]

def main_svm():
	
	M = len(all_classes)
	L = len(all_data)
	Creg = numpy.zeros((M,L))
	Cstar = numpy.zeros((M,L))

	for c_ind,class_id in enumerate(all_classes):
		for d_ind, data_id in enumerate(all_data):
			Creg[c_ind,d_ind]=numpy.loadtxt(PATH_CV_results + dataset + "_svm_" +class_id + "class_" + "%ddata_best.txt"%data_id)

	numpy.set_printoptions(precision=6)
	numpy.set_printoptions(suppress=True)
	print "Hyperparameters: SVM"
	print "Creg"
	print Creg

def main_svm_plus():

        M = len(all_classes)
        L = len(all_data)
        Creg = numpy.zeros((M,L))
        Cstar = numpy.zeros((M,L))
        for c_ind,class_id in enumerate(all_classes):
                for d_ind, data_id in enumerate(all_data):
                        Creg[c_ind,d_ind]=numpy.loadtxt(PATH_CV_results + dataset + "_svm_plus_" +class_id + "class_" + "%ddata_best.txt"%data_id)
                        Cstar[c_ind,d_ind]=numpy.loadtxt(PATH_CV_results + dataset + "_svm_plus_" +class_id + "class_" + "%ddata_best_star.txt"%data_id)
        numpy.set_printoptions(precision=6)
        numpy.set_printoptions(suppress=True)
        print "Hyperparameters: SVM+"
        print "Creg"
        print Creg
        print "Cstar"
        print Cstar


def main(method):
	
	M = len(all_classes)
	L = len(all_data)
	Creg = numpy.zeros((M,L))
	Cstar = numpy.zeros((M,L))
	
	for c_ind,class_id in enumerate(all_classes):
		for d_ind, data_id in enumerate(all_data):
			Creg[c_ind,d_ind] = numpy.loadtxt(PATH_CV_results + dataset + '_'+ method +'_'+class_id+'class_%ddata_best.txt'%data_id)
			Cstar[c_ind,d_ind]= numpy.loadtxt(PATH_CV_results + dataset + '_'+ method +'_'+class_id+'class_%ddata_best_star.txt'%data_id)		
			
	numpy.set_printoptions(precision=8)
	numpy.set_printoptions(suppress=True)
	print "Hyperparameters: ", method
	print "Creg"
	print Creg
	print "Cstar"
	print Cstar
	
def main2(method):

        M = len(all_classes)
        L = len(all_data)
        Creg = numpy.zeros((M,L))

        for c_ind,class_id in enumerate(all_classes):
                for d_ind, data_id in enumerate(all_data):
                        Creg[c_ind,d_ind] = numpy.loadtxt(PATH_CV_results + dataset + '_'+ method +'_'+class_id+'class_%ddata_best.txt'%data_id)

        numpy.set_printoptions(precision=8)
        numpy.set_printoptions(suppress=True)
        print "Hyperparameters: ", method
        print "Creg"
        print Creg
		
				
if __name__ == '__main__':

	main('privfeat_rfe_top')
	main2('privfeat_rfe_top_SVMRFE_0.05top')
	main2('svm')
