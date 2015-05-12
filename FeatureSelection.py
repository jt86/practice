__author__ = 'jt306'
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV, RFE

import time
import pdb
from Get_Full_Path import get_full_path
# import numpy as np
# import sys
# from ParamEstimation import get_gamma_from_c
import os, sys
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit
from sklearn import svm
import numpy as np
from sklearn.metrics import f1_score
from Get_Full_Path import get_full_path

class GridSeachWithCoef(GridSearchCV):
    @property
    def coef_(self):
        return self.best_estimator_.coef_


def get_best_feats(data,labels,c_values,num_folds,keyword):

    print 'beginning rfe....'
    t0 = time.clock
    parameters = {'C':c_values}#, 'gamma':get_gamma_from_c(c_values,data)}

    rfecv = RFECV(GridSeachWithCoef(SVC(kernel='linear'), parameters, cv=rs, step=0.1),)
                  # cv=StratifiedKFold(labels, num_folds, shuffle=False , random_state=1))) #,scoring='f1')# cv=StratifiedKFold(labels, init_step))

    print 'data shape',data.shape
    print 'labels shape', labels.shape
    rfecv.fit(data, labels)
    print "... RFE finished. Time to do initial feature selection:",time.clock()-t0
    print "Kept {} out of {} features".format((data[:,rfecv.support_]).shape[1], data.shape[1])
    print "support:",rfecv.support_


    return rfecv.support_




def recursive_elimination2(feats, labels, num_feats_to_select):
    print 'beginning rfe....'
    labels = np.array(labels, dtype=float)
    feats = np.array(feats, dtype=float)

    c_values = [.1, 1, 10]
    params_grid = [{'C': 0.1}, {'C': 1.}, {'C': 10.}]
#     gamma_values = get_gamma_from_c(c_values, feats)
#     params_dict = {'C': c_values, 'gamma': gamma_values}
#
#
    estimator = svm.SVC(kernel="linear", C=1)
    selector = RFECV(estimator, step=1, cv=5, n_features_to_select=num_feats_to_select)
    # selector = RFE(estimator, step=1, n_features_to_select=num_feats_to_select)
    # clf = grid_search.GridSearchCV(selector, {'estimator_params': params_grid}, cv=5)
    selector.fit(feats, labels)
    print '...finishing rfe'
    return selector.support_
#     # clf.best_estimator_.estimator_
#     # clf.best_estimator_.grid_scores_
#     ranking = clf.best_estimator_.ranking_
#
#     logging.info(ranking)
#     ranking = np.array(ranking)
#     ranking = np.subtract(ranking, 1)
#     return ranking