__author__ = 'jt306'
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import zero_one_loss
from Heart import get_heart_data
from sklearn import svm, grid_search
import time

# import numpy as np
# import sys
# from ParamEstimation import get_gamma_from_c
class GridSeachWithCoef(GridSearchCV):
    @property
    def coef_(self):
        return self.best_estimator_.coef_

# def get_best_feats(data,labels,c_values):
#
#     parameters = {'C':c_values}#, 'gamma':get_gamma_from_c(c_values,data)}
#
#     # svm1 passed to clf which is used to grid search the best parameters
#     svm1 = SVC(kernel='linear')
#     clf = grid_search.GridSearchCV(svm1, parameters, refit=True)
#     clf.fit(data,labels)
#     #print 'best gamma',clf.best_params_['gamma']
#
#     # svm2 uses the optimal hyperparameters from svm1
#     svm2 = svm.SVC(C=clf.best_params_['C'], kernel='linear')#, gamma=clf.best_params_['gamma'])
#     rfecv = RFECV(estimator=svm2, step=1, cv=StratifiedKFold(labels, 5))      #Initialise feat selection object using svc, stratified kfold
#             # ,loss_func=zero_one_loss)
#     rfecv.fit(data, labels)                                                     #Fit this feat selection object to the data
#
#     print "Kept {} out of {} features".format((data[:,rfecv.support_]).shape[1], data.shape[1])
#     print "support:",rfecv.support_
#     return data[:,rfecv.support_]


def get_best_feats(data,labels,c_values):
    t0 = time.clock()

    parameters = {'C':c_values}#, 'gamma':get_gamma_from_c(c_values,data)}

    rfecv = RFECV(GridSeachWithCoef(SVC(kernel='linear'), parameters),step=1, cv=StratifiedKFold(labels, 5))
    rfecv.fit(data, labels)

    print "Kept {} out of {} features".format((data[:,rfecv.support_]).shape[1], data.shape[1])
    print "support:",rfecv.support_
    print "time to do initial feature selection:",time.clock()-t0
    return data[:,rfecv.support_]


# X,y = get_heart_data()
#
# c_values = [0.1,1.,10.]
# get_best_feats2(X,y,c_values)


