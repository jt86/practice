__author__ = 'jt306'
import numpy as np
from cvxopt import matrix, solvers

n = 5

# G = np.zeros((2*n, 2*n))
# # print G
#
# G[0:n,0:n] = np.eye(n)
# # print G
#
#
#
# G[n:2*n,0:n] = np.diag(np.r_[[-1.0]*n])
# # print G
# #
# #
# #
# G[n:2*n,n:2*n] = np.eye(n)
# print G
#
#
# C=1
# h = np.zeros((2*n,1))  #matrix(0.0, (2*n,1))
# h[n:2*n,0] = np.r_[[C]*n]
# print h
#
#
# Q = 2*matrix([ [2, .5], [.5, 1] ])
# p = matrix([1.0, 1.0])
# G = matrix([[-1.0,0.0],[0.0,-1.0]])
# h = matrix([0.0,0.0])
# A = matrix([1.0, 1.0], (1,2))
# b = matrix(1.0)
#
# print "Q", Q
# print "p",p
# print "G",G
# print "h",h
# print "A",A
# print "b",b
#
# print ""

P = np.zeros((2*n,2*n)) #matrix(0.0,(2*n,2*n))
P[0:n,0:n] = K*np.dot(Y,Y.T)
P[n:2*n,n:2*n] = Kstar/Cstar
P = P + np.eye(2*n)*1.e-10

Q = matrix(0.0, (2*n,1))
Q[0:n] = np.r_[[-1.0]*n]
A = np.zeros((2,2*n)) #matrix(0.0, (2,2*n))
b = np.zeros((2,1)) #matrix(0.0, (2,1))
A[0,n:2*n] = np.r_[[1.0]*n]

print P
print Q