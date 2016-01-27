__author__ = 'jt306'
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn import feature_selection as feature_selection
iris = load_iris()
X, y = iris.data, iris.target
X.shape

print (sklearn.__version__)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = feature_selection.SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape

# data = np.random.rand(4,4)
# print(data)
# print ('\n\n')
# normalised = preprocessing.normalize(data, norm='l1', axis=1, copy=True)
#
# print (normalised)
# print (np.sum(normalised, axis=1))