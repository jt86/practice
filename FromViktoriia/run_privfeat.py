#!/usr/bin/python
import sys
import numpy
import pdb
from sklearn import cross_validation, linear_model
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from getdata import getdata_AwA_one_vs_rest, getdata_arcene 
from getmethod import do_CV_svm_plus_5x5fold, do_CV_svm_5x5fold
from getmethod import do_CV_svmrfe_5fold, do_CV_svml1_5fold, do_CV_logreg_5fold
from SVMplus import svmplusQP, svmplusQP_Predict

from Get_Full_Path import get_full_path
from Get_Full_Path import get_full_path

#PATH = "/home/n/nq/nq28/PrivFeatures/"
# PATH = '/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/FromViktoriia'

PATH=''
def main():
    k = eval(sys.argv[1])
    class_id = sys.argv[2]		#class_id depending on the dataset AwA:'0','1',...,'9'; arcene:'01'
    N = eval(sys.argv[3])		#N depending on the dataset
    test_N=eval(sys.argv[4])	#test_N depending on the dataset
    method = sys.argv[5]
    topK=eval(sys.argv[6])		#top features(fraction of features) to select for privfeat_rfe; for example, 0.05, 0.1, 0.25, 0.5, 0.75
    dataset = sys.argv[7]

    print method+" method; "+dataset+" dataset; "+class_id+" class id; "+"%d"%k+" repeat; "+"N=%d number of positive samples;"%N+" Top %.2ffeatures"%topK

    PATH_data = PATH + dataset + "/data_" + dataset + "/"
    PATH_CV_results = PATH + dataset + "/CV/"
    PATH_results = PATH + dataset + "/Results/"

    numpy.random.seed(k)
    if (dataset == 'AwA'):
        (data, test_data, Y, test_Y) = getdata_AwA_one_vs_rest(PATH_data, class_id, N, test_N)
        reg_array= [1.0, 10., 100., 1000., 10000.]

    if (dataset == 'arcene'):
        (data, test_data, Y, test_Y) = getdata_arcene(PATH_data, class_id, N, test_N)
        reg_array= [1.0, 10., 100., 1000., 10000., 100000., 1000000., 10000000.]

    top=int(topK*data.shape[1])	#top features to select

    if (method == 'privfeat_rfe_top'):

        #feature selection with RFE with "top" features to select
        reg_best = 10.#do_CV_svmrfe_5fold(data, Y, reg_array, dataset, PATH_CV_results, method + "_SVMRFE_%.2ftop"%topK, class_id, k,top)
        #reg_best=numpy.loadtxt(PATH_CV_results + dataset + "_" + method + "_SVMRFE_%.2ftop"%topK+ "_" +class_id + "class_"+ "%ddata_best.txt"%k)
        print "svm+rfe Regularization:", reg_best

        svc = SVC(C=reg_best, kernel="linear", random_state=1)
        rfe = RFE(estimator=svc, n_features_to_select=top, step=1)
        rfe.fit(data, Y)
        ACC = rfe.score(test_data, test_Y)
        selected = rfe.support_
        notselected = numpy.invert(rfe.support_)

        filename=PATH_results+dataset+"_"+method
        numpy.savetxt(filename + "_SVMRFE_%.2ftop_"%topK+class_id+"class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]),fmt='%f')
        print 'Accuracy at selecting: ', ACC
        #SVM+ part
        X_selected=data[:,selected].copy();
        test_X_selected=test_data[:,selected].copy()
        X_priv=data[:,notselected].copy()
        #(reg_best, reg_best_star) = do_CV_svm_plus_5x5fold(X_selected, Y, X_priv, reg_array, reg_array, dataset, PATH_CV_results, method+"_%.2ftop"%topK, class_id, k)
        reg_best=100.
        reg_best_star=1.
        duals,bias = svmplusQP(X_selected,Y.copy(),X_priv,reg_best,reg_best_star)
        testXranked = svmplusQP_Predict(X_selected,test_X_selected,duals,bias).flatten()
        ACC = numpy.sum(test_Y==numpy.sign(testXranked))/(1.*len(test_Y))

        numpy.savetxt(filename + "_%.2ftop_"%topK + class_id + "class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]), fmt='%f')
        print 'Privileged selection: ', ACC

    #baselines that do not depend on parameter top================================================================
    if (method == 'privfeat_svml1'):

        #feature selection with L1 regularized SVM with squared hinge loss
        reg_best = do_CV_svml1_5fold(data,Y, reg_array, dataset, PATH_CV_results, method + "_L1SVM", class_id, k)
        print "L1svm, Regularization:", reg_best

        svc = LinearSVC(C=reg_best, penalty="l1", dual=False, random_state=1)
        svc.fit(data, Y)
        ACC = svc.score(test_data, test_Y)
        model = svc.coef_.reshape(-1,)
        selected = model!=0
        notselected = model==0
        idx=(numpy.arange(data.shape[1]))[selected]

        numpy.savetxt(PATH_results+dataset+"_"+method+ "_L1SVM_" +class_id + "class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]), fmt='%f')
        numpy.savetxt(PATH_results+dataset+"_"+method+ "_" +class_id + "class_"+ "%ddata_selectedfeatidx.txt"%k, numpy.asarray(idx), fmt='%d')
        print 'Accuracy at selecting: ', ACC

        #SVM+ part
        X_selected=data[:,selected].copy();
        test_X_selected=test_data[:,selected].copy()
        X_priv=data[:,notselected].copy()

        (reg_best, reg_best_star) = do_CV_svm_plus_5x5fold(X_selected, Y, X_priv, reg_array, reg_array, dataset, PATH_CV_results, method, class_id, k)
        duals,bias = svmplusQP(X_selected,Y.copy(),X_priv,reg_best,reg_best_star)
        testXranked = svmplusQP_Predict(X_selected,test_X_selected,duals,bias).flatten()
        ACC = numpy.sum(test_Y==numpy.sign(testXranked))/(1.*len(test_Y))

        numpy.savetxt(PATH_results + dataset + "_" + method +"_" + class_id + "class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]), fmt='%f')
        print 'Privileged selection: ', ACC

    if (method == 'privfeat_logreg'):
        ##feature selection with L1 regularized logistic regression
        reg_best = do_CV_logreg_5fold(data,Y, reg_array, dataset, PATH_CV_results, method + "_L1LR", class_id, k)
        print "Logreg, Regularization:", reg_best

        svc = linear_model.LogisticRegression(C=reg_best, penalty="l1", dual=False, random_state=1)
        svc.fit(data, Y)
        ACC = svc.score(test_data, test_Y)
        model = svc.coef_.reshape(-1,)
        selected = model!=0
        notselected = model==0
        idx=(numpy.arange(data.shape[1]))[selected]

        numpy.savetxt(PATH_results+dataset+"_"+method+ "_L1LR_" +class_id + "class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]), fmt='%f')
        numpy.savetxt(PATH_results+dataset+"_"+method+ "_" +class_id + "class_"+ "%ddata_selectedfeatidx.txt"%k, numpy.asarray(idx), fmt='%d')
        print 'Accuracy at selecting: ', ACC

        #SVM+ part
        X_selected=data[:,selected].copy();
        test_X_selected=test_data[:,selected].copy()
        X_priv=data[:,notselected].copy()
        (reg_best, reg_best_star) = do_CV_svm_plus_5x5fold(X_selected, Y, X_priv, reg_array, reg_array, dataset, PATH_CV_results, method, class_id, k)

        duals,bias = svmplusQP(X_selected,Y.copy(),X_priv,reg_best,reg_best_star)
        testXranked = svmplusQP_Predict(X_selected,test_X_selected,duals,bias).flatten()
        ACC = numpy.sum(test_Y==numpy.sign(testXranked))/(1.*len(test_Y))

        numpy.savetxt(PATH_results + dataset + "_" + method +"_" + class_id + "class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]), fmt='%f')
        print 'Privileged selection: ', ACC

    if (method == 'svm'):
        #SVM baseline on data
        reg_best = do_CV_svm_5x5fold(data,Y, reg_array, dataset, PATH_CV_results, method, class_id, k)
        print "Regularization:", reg_best

        svc = SVC(C=reg_best, kernel="linear", random_state=1)
        svc.fit(data, Y)
        ACC = svc.score(test_data, test_Y)
        numpy.savetxt(PATH_results+dataset+"_"+method+ "_" + class_id + "class_"+ "%ddata_ACC.txt"%k, numpy.asarray([[ACC]]), fmt='%f')
        print 'SVM accuracy: ', ACC

if __name__ == '__main__':
    main()

