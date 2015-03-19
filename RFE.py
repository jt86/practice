__author__ = 'jt306'
from ParamEstimation import get_gamma_from_c
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import numpy as np
from sklearn import grid_search
from Heart import get_heart_data


X, Y = get_heart_data()
# X = np.array([5,4,3,2,1]*5+[1,2,3,4,5]*5, )
# Y = [1]*6+[-1]*4

# X.shape = (10,5)

# def recursive_elimination2(feats,labels,num_feats_to_select):
#     c_values = [0.1, 1., 10.]
#     gamma_values = get_gamma_from_c(c_values,feats)
#     params_dict = {'C':c_values}#, 'Gamma':gamma_values}
#
#     # logging.info( 'shape',feats.shape[1])
#     if feats.shape[1]>100:
#         step = 0.1
#     else:
#         step = 1
#     # logging.info( 'num feats eliminated each time', step)
#     svc = SVC(kernel="linear", C=1)
#     rfe = RFE(estimator=svc, n_features_to_select=num_feats_to_select, step=step, estimator_params=params_dict)
#     rfe.fit(feats, labels)
#     ranking = np.subtract(rfe.ranking_.reshape(len(feats[0])),1)
#     grid_search_object = grid_search.GridSearchCV(estimator = rfe)
#     return ranking
#
# logging.info( recursive_elimination2(X,Y,1))
# w

def grid_search_svc():
    svr = SVC()
    c_values = [.1, 1, 10]
    gamma_values = get_gamma_from_c(c_values, X)
    params_dict = {'C': c_values, 'gamma': gamma_values}
    grid_search_clf = grid_search.GridSearchCV(svr, params_dict, n_jobs=4)
    grid_search_clf.fit(X, Y)

    # logging.info( grid_search_clf.grid_scores_)
    # logging.info( grid_search_clf.score(X, Y))
    # logging.info( grid_search_clf.best_params_)

    return grid_search_clf.best_params_

# feat_eliminator = RFE()
# grid_search_clf = grid_search.GridSearchCV(svr, params_dict, n_jobs=4)