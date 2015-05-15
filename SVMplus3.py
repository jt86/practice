# QP solution of SVM+
# based on Fast Optimization Algorithms for Solving SVM+ (D. Pechyony and V. Vapnik)
# 22 July 2014
# Questions regarding this code are directed to: N.Quadrianto@sussex.ac.uk
from __future__ import division
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
from cvxopt import solvers
solvers.options['show_progress'] = False
from vector import CGaussKernel,CLinearKernel,CRBFKernel
import numpy as np
import pdb
import numpy.random as random
import logging

def svmplusQP(X,Y,Xstar,C,Cstar, gamma=None, gammastar=None):
    n = X.shape[0]
    Y.shape = n,1

    # If gamma is passed, is using rbf so
    if gamma!=None:
    # Compute kernel matrices
        dk = CRBFKernel();
        dkstar = CRBFKernel();
        dK = dk.Dot(X, X)
        dKstar = dkstar.Dot(Xstar, Xstar)

        # omega_K = 1.0 / np.median(dK.flatten())                   #todo: these two lines modified for param estimation
        # omega_Kstar = 1.0 / np.median(dKstar.flatten())
        omega_K = gamma
        omega_Kstar = gammastar


        kernel_K = CGaussKernel(omega_K) #CLinearKernel()
        kernel_Kstar = CGaussKernel(omega_Kstar) # CLinearKernel()

    else:
        print 'linear kernel for svm plus!'
        kernel_K = CLinearKernel()
        kernel_Kstar = CLinearKernel()



    K = kernel_K.Dot(X,X)
    Kstar = kernel_Kstar.Dot(Xstar,Xstar)


    P = np.zeros((2*n,2*n)) #matrix(0.0,(2*n,2*n))
    P[0:n,0:n] = K*np.dot(Y,Y.T)
    P[n:2*n,n:2*n] = Kstar/Cstar
    P = P + np.eye(2*n)*1.e-10

    Q = matrix(0.0, (2*n,1))
    Q[0:n] = np.r_[[-1.0]*n]
    A = np.zeros((2,2*n)) #matrix(0.0, (2,2*n))
    b = np.zeros((2,1)) #matrix(0.0, (2,1))
    A[0,n:2*n] = np.r_[[1.0]*n]

    A[1,0:n] = Y.flatten()


    G = np.zeros((2*n,2*n)) #matrix(0.0, (2*n,2*n))
    G[0:n,0:n] = np.eye(n)
    G[n:2*n,0:n] = np.diag(np.r_[[-1.0]*n])
    G[n:2*n,n:2*n] = np.eye(n)


    h = np.zeros((2*n,1))  #matrix(0.0, (2*n,1))
    h[n:2*n,0] = np.r_[[C]*n]

    sol = qp(matrix(P), matrix(Q), matrix(-1.0*G), matrix(h), matrix(A), matrix(b))['x']
    sol = np.array(sol)

    alphas = sol[0:n]
    betahats = sol[n:2*n]
    betas = betahats - alphas + C



    alphas[ np.abs(alphas) < 1e-5 ] = 0
    betas[ np.abs(betas) < 1e-5 ] = 0

    # We compute the bias as explained in Fast Optimization Algorithms for Solving SVM+ (Section 1.5.1)


    Fi = np.dot(K,Y*alphas)
    fi = np.dot(Kstar,betahats)

    rangeobs = range(n)


    sel_pos = ((alphas.flatten() > 0) & (Y.flatten()==1))
    sel_neg = ((alphas.flatten() > 0) & (Y.flatten()==-1))




    if (sel_pos.shape[0] > 0):
        s_pos = np.mean((1 - fi / Cstar - Fi)[ sel_pos ])

    else:
        s_pos = 0

    if (sel_neg.shape[0] > 0):
        s_neg = np.mean((-1 + fi / Cstar - Fi)[ sel_neg ])

    else:
        s_neg = 0



    bias = (s_pos + s_neg) / 2.



    return alphas*Y,bias



def svmplusQP_Predict(X,Xtest,alphas,bias, kernel):

    if kernel=='rbf':
        # Compute kernel matrices
        dk = CRBFKernel();
        dK = dk.Dot(X, Xtest)
        omega_K = 1.0 / np.median(dK.flatten())
        kernel_K = CGaussKernel(omega_K) # CLinearKernel()
    else:
        kernel_K =  CLinearKernel()

    K = kernel_K.Dot(X,Xtest)
    predicted = np.dot(K.T,alphas)+bias
    return np.sign(predicted)

if __name__ == "__main__":
    X = random.randn(1000,999)
    Xtest = random.randn(200,999)
    Xstar = random.randn(1000,90)
    Y = np.r_[[1]*500, [-1]*500]
    C = 0.1
    Cstar = 0.1
    duals,bias = svmplusQP(X,Y,Xstar,C,Cstar)
    predicted = svmplusQP_Predict(X,Xtest,duals,bias)

    print(predicted)

#
#def
#(X, Y, Xstar, Xtest, C, Cstar):
#    duals,bias = svmplusQP(X, Y, Xstar, C, Cstar)
#    return svmplusQP_Predict(X, Xtest, duals, bias)