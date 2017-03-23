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

## Unit tests for all vector kernels.

import time
import numpy
import numpy.random as random
import unittest

import vector
from bahsic import CBAHSIC

bahsic = CBAHSIC()


## Class that tests BAHSIC with and without optimization.
#
class CTestBAHSIC(unittest.TestCase):

    ## Test routine for BAHSIC without optimization in linear problem.
    # Test CBAHSIC.BAHSICRaw
    # 
    def testBAHSICLinear(self):
        print('')
        print('###########################################################')
        print('# Linear problem: only the first 8 dimension are relevant #')
        print('###########################################################')
        data_no = 100
        dim_no = 100

        # Generate the data.
        data = random.rand(data_no, dim_no)
        data[:,3] = data[:,2] + 0.1*random.randn(data_no)
        data[:,4] = data[:,1] + 0.1*random.randn(data_no)
        data[:,5:9] = data[:,1:5] + 0.1*random.randn(data_no, 4)

        # Generate the labels.
        y = 2*(numpy.ceil(data[:,1]+data[:,2]-1) - 0.5)
        y.shape = (data_no,1)

        # Normalize the labels.
        y = 1.0*y
        tmp_no = numpy.sum(y)
        pno = (data_no + tmp_no) / 2
        nno = (data_no - tmp_no) / 2
        y[y>0] = y[y>0]/pno
        y[y<0] = y[y<0]/nno

        # Normalize the data. 
        m = data.mean(0)
        s = data.std(0)
        data.__isub__(m).__idiv__(s)
        
        print('data no:', data_no, 'dimension no:', \
              dim_no, 'sample no:', pno, ' vs.', nno)

        print('')
        print('--Linear kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICRaw(data, y, vector.CLinearKernel(), \
                               vector.CLinearKernel(), 8, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

        print('')
        print('--Inverse distance kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICRaw(data, y, vector.CInvDisKernel(), \
                               vector.CLinearKernel(), 8, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

        print('')
        print('--Gauss kernel with scale parameter 10 on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICRaw(data, y, vector.CGaussKernel(10), \
                               vector.CLinearKernel(), 8, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

        print('')
        print('--Gauss kernel with scale parameter 0.1 on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICRaw(data, y, vector.CGaussKernel(0.1), \
                               vector.CLinearKernel(), 8, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

        print('here')

    ## Test routine for BAHSIC without optimization in nonlinear problem.
    # Test CBAHSIC.BAHSICRaw
    #
    def testBAHSICNonlinear(self):
        print('')
        print('#############################################################')
        print('# Nonlinear XOR problem: the first 2 dimension are relevant #')
        print('#############################################################')
        data_no = 100
        dim_no = 20

        # Generate the data.
        data = random.rand(data_no, dim_no)
        # Generate the labels.
        y = 2*numpy.bitwise_xor( ((numpy.sign(data[:,1]-0.5)+1)/2).astype(numpy.int32)\
                                 ,((numpy.sign(data[:,2]-0.5)+1)/2).astype(numpy.int32))\
                                 - 1
        y.shape = (data_no,1)

        # Normalize the labels.
        y = 1.0*y
        tmp_no = numpy.sum(y)
        pno = (data_no + tmp_no) / 2
        nno = (data_no - tmp_no) / 2
        y[y>0] = y[y>0]/pno
        y[y<0] = y[y<0]/nno

        # Normalize the data.
        m = data.mean(0)
        s = data.std(0)
        data.__isub__(m).__idiv__(s)
        
        print('data no:', data_no, 'dimension no:', dim_no, 'sample no:', \
              pno, ' vs.', nno)

        print('')
        print('--Linear kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICRaw(data, y, vector.CLinearKernel(), \
                               vector.CLinearKernel(), 2, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)
        
        print('')
        print('--Inverse distance kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICRaw(data, y, vector.CInvDisKernel(), \
                               vector.CLinearKernel(), 2, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

    ## Test routine for BAHSIC with optimization in linear problem.
    # Test CBAHSIC.BAHSICOpt
    #
    def testBAHSICOptLinear(self):
        print('')
        print('###########################################################')
        print('# Linear problem: only the first 8 dimension are relevant #')
        print('###########################################################')
        data_no = 100
        dim_no = 20

        # Generate the data.
        data = random.rand(data_no, dim_no)
        data[:,3] = data[:,2] + 0.1*(random.rand(data_no)-0.5)
        data[:,4] = data[:,1] + 0.1*(random.rand(data_no)-0.5)
        data[:,5:9] = data[:,1:5] + 0.1*(numpy.reshape(random.rand(data_no*4),(data_no,4))-0.5)
        # Generate the labels.
        y = 2*(numpy.ceil(data[:,1]+data[:,2]-1) - 0.5)

        # Normalize the labels.
        y = 1.0*y
        tmp_no = numpy.sum(y)
        pno = (data_no + tmp_no) / 2
        nno = (data_no - tmp_no) / 2
        y[y>0] = y[y>0]/pno
        y[y<0] = y[y<0]/nno
        y.shape = (data_no,1)

        # Normalize the data.
        m = data.mean(0)
        s = data.std(0)
        data.__isub__(m).__idiv__(s)
        
        print('data no:', data_no, 'dimension no:', dim_no, 'sample no:', \
              pno, ' vs.', nno)

        print('')
        print('--Linear kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICOpt(data, y, vector.CLinearKernel(), \
                               vector.CLinearKernel(), 8, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

        print('')
        print('--RBF kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICOpt(data, y, vector.CLaplaceKernel(), \
                               vector.CLinearKernel(), 8, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

    ## Test routine for BAHSIC with optimization in nonlinear problem.
    # Test CBAHSIC.BAHSICOpt
    #
    def testBAHSICOptNonlinear(self):
        print('')
        print('#############################################################')
        print('# Nonlinear XOR problem: the first 2 dimension are relevant #')
        print('#############################################################')
        data_no = 100
        dim_no = 20

        # Generate the data.
        data = random.rand(data_no, dim_no)
        # Generate the labels.
        y = 2*numpy.bitwise_xor( ((numpy.sign(data[:,1]-0.5)+1)/2).astype(numpy.int32)\
                              ,((numpy.sign(data[:,2]-0.5)+1)/2).astype(numpy.int32))\
                              - 1

        # Normalize the labels.
        y = 1.0*y
        tmp_no = numpy.sum(y)
        pno = (data_no + tmp_no) / 2
        nno = (data_no - tmp_no) / 2
        y[y>0] = y[y>0]/pno
        y[y<0] = y[y<0]/nno
        y.shape = (data_no,1)

        # Normalize the data.
        m = data.mean(0)
        s = data.std(0)
        # data.__isub__(m).__idiv__(s)
        data-=m
        data/=s

        print('data no:', data_no, 'dimension no:', dim_no, 'sample no:', \
              pno, ' vs.', nno)

        print('')
        print('--Linear kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICOpt(data, y, vector.CLinearKernel(), \
                               vector.CLinearKernel(), 2, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)

        print('')
        print('--RBF kernel on data:')
        t1 = time.clock()
        tmp = bahsic.BAHSICOpt(data, y, vector.CLaplaceKernel(), \
                               vector.CLinearKernel(), 2, 0.1)
        t2 = time.clock()
        print(t2-t1)
        print('--rank of the features')
        print('--better features towards the end of the list:')
        print(tmp)
        print('there')

if __name__== '__main__':
    unittest.main()
