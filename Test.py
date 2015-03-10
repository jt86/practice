__author__ = 'jt306'
from SVMplus import svmplusQP, svmplusQP_Predict
import numpy as np


X = [5,4,3,2,1]*5+[1,2,3,4,5]*5
Y = [1]*5+[-1]*5

X=np.array(X)
Y=np.array(Y)

X.shape = (10,5)
Y.shape= (10,1)

print X
print Y

Xstar = np.array([[]]*10)
print Xstar.shape
Xstar.shape=(10,0)
print Xstar


C, Cstar = 10,10
gamma, gammastar = 0.1, 0.1
 #


Xtest = np.array([5,5,3,1,1]*4)
Xtest.shape = (4,5)

alphas,bias=svmplusQP(X,Y,Xstar,C,Cstar,gamma,gammastar)
print (svmplusQP_Predict(X,Xtest,alphas,bias))