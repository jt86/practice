__author__ = 'jt306'
import numpy as np
from sklearn.preprocessing import Imputer


X = [5,4,3,2,1]*5+[4,3,2,1,0]*5+['NaN',2,3,'NaN',5]
Y = [1]*5+[-1]*5

X=np.array(X)
Y=np.array(Y)

X.shape = (11,5)
Y.shape= (10,1)

# logging.info( X)


def imputer(X):
 # missing_values is the value of your placeholder, strategy is if you'd like mean, median or mode, and axis=0 means it calculates the imputation based on the other feature values for that sample
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
    train_imp = imp.transform(X)
    return train_imp

# logging.info( imputer(X))