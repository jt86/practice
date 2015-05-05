__author__ = 'jt306'
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import zero_one_loss
from Heart import get_heart_data
from sklearn import svm, grid_search
import time
import pdb
from Get_Full_Path import get_full_path
# import numpy as np
# import sys
# from ParamEstimation import get_gamma_from_c
import os, sys
from sklearn.cross_validation import StratifiedKFold,KFold, ShuffleSplit
import numpy as np
from sklearn.metrics import f1_score
from Get_Full_Path import get_full_path

class GridSeachWithCoef(GridSearchCV):
    @property
    def coef_(self):
        return self.best_estimator_.coef_


def get_best_feats(data,labels,c_values,num_folds,rs,keyword):
    t0 = time.clock()
    if data.shape[1] > 100:
        step = 0.01
    else:
        step = 1
    parameters = {'C':c_values}#, 'gamma':get_gamma_from_c(c_values,data)}
    rfecv = RFECV(GridSeachWithCoef(SVC(kernel='linear'), parameters, cv=rs),step=step,
                  cv=StratifiedKFold(labels, num_folds, shuffle=False , random_state=1))#,scoring='f1')# cv=StratifiedKFold(labels, init_step))
    print 'data shape',data.shape
    print 'labels shape', labels.shape

    rfecv.fit(data, labels)
    # pdb.set_trace()
    print "Kept {} out of {} features".format((data[:,rfecv.support_]).shape[1], data.shape[1])
    print "support:",rfecv.support_
    print "time to do initial feature selection:",time.clock()-t0

    best_feats_doc = open('best_feats.csv','a')


    directory = get_full_path('Desktop/Privileged_Data/bestfeats')
    if not os.path.exists(directory):
        os.mkdir(directory)
    best_feats_doc = open(os.path.join(directory, keyword),'a')
    best_feats_doc.write(rfecv.support_)
    best_feats_doc.write(rfecv.support_==False)

    return rfecv.support_, rfecv.support_==False



