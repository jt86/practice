# Copyright (c) 2006, National ICT Australia
# All rights reserved.
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the 'License'); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an 'AS IS' basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
# Authors: Le Song (lesong@it.usyd.edu.au)
# Created: (20/10/2006)
# Last Updated: (dd/mm/yyyy)
#

##\package elefant.fselection.bahsic
# This module perform backward elimination for feature selection
# using HSIC (BAHSIC).
#
# The algorithm proceeds recursively, eliminating the least
# relevant features and adding them to the eliminated list
# in each iteration. For more theoretical underpinning see the
# following reference for more information:
#
# Le Song, Justin Bedo, Karsten M. Borgwardt, Arthur Gretton
# and Alex Smola. The BAHSIC family of gene selection algorithms.
#

__version__ = '$Revision: $' 
# $Source$

from Get_Full_Path import get_full_path
import numpy
from scipy import optimize

import vector
from hsic import CHSIC
from setdiag0 import setdiag0


## Class that perform backward elimination for feature selection (BAHSIC).
#
# It has two version of BAHSIC: one without optimization over the kernel
# parameters and one with optimization over the kernel parameters.
#
class CBAHSIC(object):
    def __init__(self):
        pass

    ## BAHSIC with optimization over the kernel parameters.
    # @param x The data.
    # @param y The labels.
    # @param kernelx The kernel on the data.
    # @param kernely The kernel on the labels.
    # @param flg3 The number of desired features.
    # @param flg4 The proportion of features eleminated in each iteration.
    #
    def BAHSICOpt(self, x, y, kernelx, kernely, flg3, flg4):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y.shape) == 2, 'Argument 2 has wrong shape'
        assert x.shape[0] == y.shape[0], \
               'Argument 1 and 2 have different number of data points'
                       
        print ('--initializing...')
        hsic = CHSIC()
        
        L = kernely.Dot(y, y)
        setdiag0(L)
        sL = numpy.sum(L, axis=1)
        ssL = numpy.sum(sL)

        n = x.shape
        eliminatedI = []
        selectedI = set(numpy.arange(n[1]))

        kernelx.CreateCacheKernel(x)
        sga = kernelx._typicalParam
        sgaN = sga.shape
        sgaN = sgaN[0]

        while True:        
            selectedI = selectedI - set(eliminatedI)
            sI = numpy.array([j for j in selectedI])
            m = len(sI)

            print (m)
            if (m == 1):
                eliminatedI.append(selectedI.pop())
                break

            sgaMat = []
            hsicMat = []
            for k in range(sgaN):
                ## bfgs in scipy is not working here
                retval = optimize.fmin_cg(hsic.ObjUnBiasedHSIC, \
                                          sga[[k],].ravel(), \
                                          hsic.GradUnBiasedHSIC,\
                                          args=[x, kernelx, L, sL, ssL], \
                                          gtol=1e-6, maxiter=100, \
                                          full_output=True, disp=False)
                sgaMat.append(retval[0])
                hsicMat.append(retval[1])
                    
            k = numpy.argmin(hsicMat)
            sga0 = sgaMat[k]
            
            objj = []
            for j in selectedI:
                K = kernelx.DecDotCacheKernel(x, x[:,[j]], sga0)
                setdiag0(K)
                objj.append(hsic.UnBiasedHSICFast(K, L, sL, ssL))

            if m > flg3:
                maxj = numpy.argsort(objj)
                num = int(flg4 * m)+1
                if m - num <= flg3:
                    num = m - flg3
                maxj = maxj[m:m-num-1:-1]
            else:
                maxj = numpy.array([numpy.argmax(objj)])
                
            j = numpy.take(sI,maxj)
            eliminatedI.extend(j)
            kernelx.DecCacheKernel(x, x[:,j])

        kernelx.ClearCacheKernel(x)
        return eliminatedI

    ## BAHSIC without optimization over the kernel parameters.
    # @param x The data.
    # @param y The labels.
    # @param kernelx The kernel on the data.
    # @param kernely The kernel on the labels.
    # @param flg3 The number of desired features.
    # @param flg4 The proportion of features eleminated in each iteration.
    #
    def BAHSICRaw(self, x, y, kernelx, kernely, flg3, flg4):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y.shape) == 2, 'Argument 2 has wrong shape'
        assert x.shape[0] == y.shape[0], \
               'Argument 1 and 2 have different number of data points'       

        print ('--initializing...')
        hsic = CHSIC()

        L = kernely.Dot(y, y)
        setdiag0(L)

        sL = numpy.sum(L, axis=1)
        ssL = numpy.sum(sL)

        n = x.shape
        eliminatedI = []
        selectedI = set(numpy.arange(n[1]))

        kernelx.CreateCacheKernel(x)

        while True:
            selectedI = selectedI - set(eliminatedI)
            sI = numpy.array([j for j in selectedI])
            m = len(sI)

            print (m)
            if (m == 1):
                eliminatedI.append(selectedI.pop())
                break

            objj = []
            for j in selectedI:
                K = kernelx.DecDotCacheKernel(x, x[:,[j]])
                setdiag0(K)
                objj.append(hsic.UnBiasedHSICFast(K, L, sL, ssL))

            if m > flg3:
                maxj = numpy.argsort(objj)
                num = int(flg4 * m)+1
                if m-num <= flg3:
                    num = m - flg3
                maxj = maxj[m:m-num-1:-1]
            else:
                maxj = numpy.array([numpy.argmax(objj)])

            j = numpy.take(sI,maxj)
            eliminatedI.extend(j)
            kernelx.DecCacheKernel(x, x[:,j])

        kernelx.ClearCacheKernel(x)
        return eliminatedI

from GetSingleFoldData import get_train_and_test_this_fold
import numpy as np
# all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold('tech',1,1,1)
# x = np.vstack((all_training,all_testing))
# y = np.hstack((training_labels,testing_labels))
# y = np.reshape(y,[y.shape[0],1])
# print(y.shape)
#
# x  = np.load('../SVMdelta/Data/Dataset219/tech219-0-0-train_priv.npy')
# y  = np.load('../SVMdelta/Data/Dataset219/tech219-0-0-train_labels.npy')
# y = np.reshape(y,[y.shape[0],1])

# print(x.shape)
# import sys
# sys.exit()
cbahsic= CBAHSIC()

## BAHSIC with optimization over the kernel parameters.
# @param x The data.
# @param y The labels.
# @param kernelx The kernel on the data.
# @param kernely The kernel on the labels.
# @param flg3 The number of desired features.
# @param flg4 The proportion of features eleminated in each iteration.
#

# output1 =((cbahsic.BAHSICOpt(x=x, y=y, kernelx=vector.CLinearKernel(), kernely=vector.CLinearKernel(), flg3=100, flg4=0.5)))
# print(output1)
# print(len(output1))
# np.save(get_full_path('Desktop/output1'),output1)
# +
# output2 =((cbahsic.BAHSICOpt(x=x, y=y, kernelx=vector.CLinearKernel(), kernely=vector.CLinearKernel(), flg3=50, flg4=0.5)))
# print(output2)
# print(len(output2))
# np.save(get_full_path('Desktop/output2'),output2)
#
# output3 =((cbahsic.BAHSICOpt(x=x, y=y, kernelx=vector.CLinearKernel(), kernely=vector.CLinearKernel(), flg3=10, flg4=0.5)))
# print(output3)
# print(len(output3))
# np.save(get_full_path('Desktop/output3'),output3)

#
# output1 = np.load(get_full_path('Desktop/output1.npy'))
# output2 = np.load(get_full_path('Desktop/output2.npy'))
# output3 = np.load(get_full_path('Desktop/output3.npy'))
# i=0
# for count,(item1,item2,item3) in enumerate(zip(output1,output2,output3)):
#     if len(set([item1, item2, item3]))!=1: #check all different
#         print([item1, item2, item3], set([item1, item2, item3]), count)
#     # print (count, item1,item2,item3,('{}'.format('<----------')) if item1 !=item3)
#                                                                     #or item2 !=item3 or item1 != item3
#
#
# print(i)


# from sklearn.svm import SVC
#
# first_100 = output1[:1000]
# last_100 = output1[-1000:]
# svm = SVC()
# svm.fit(X=x[:150,first_100],y=y[:150])
# print(svm.score(x[150:,first_100],y[150:]))
#
# svm = SVC()
# svm.fit(X=x[:150,last_100],y=y[:150])
# print(svm.score(x[150:,last_100],y[150:]))
