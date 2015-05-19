__author__ = 'jt306'
import os, sys
import argparse

from Arcene import get_arcene_data
from Gisette import get_gisette_data
from Get_Full_Path import get_full_path
from Madelon import get_madelon_data
from Dorothea import get_dorothea_data
from Vote import get_vote_data
from Heart import get_heart_data
from Haberman import get_haberman_data
from Crx import get_crx_data
from Mushroom import get_mushroom_data
from Hepatitis import get_hepatitis_data
from Cancer import get_cancer_data
from Bankruptcy import get_bankruptcy_data
from Spambase import get_spambase_data
from Musk2 import get_musk2_data
from Musk1 import get_musk1_data
from Ionosphere import get_ionosphere_data
from HillValley import get_hillvalley_data
from Wine import get_wine_data
from ParamEstimation2 import get_gamma_from_c
from sklearn import svm, grid_search
from sklearn.feature_selection import RFECV
import logging

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from Heart import get_heart_data
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import numpy as np


#
# def recursive_elimination(feats, labels):
#     svc = SVC(kernel="linear", C=1)
#     rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
#     rfe.fit(feats, labels)
#     ranking = np.subtract(rfe.ranking_.reshape(len(feats[0])), 1)
#     return ranking


# def recursive_elimination2(feats, labels, num_feats_to_select):
#     print 'beginning rfe....'
#     labels = np.array(labels, dtype=float)
#     feats = np.array(feats, dtype=float)
#
#     c_values = [.1, 1, 10]
#     params_grid = [{'C': 0.1}, {'C': 1.}, {'C': 10.}]
# #     gamma_values = get_gamma_from_c(c_values, feats)
# #     params_dict = {'C': c_values, 'gamma': gamma_values}
# #
# #
#     estimator = svm.SVC(kernel="linear")
#     selector = RFECV(estimator, step=1, cv=5, n_features_to_select=num_feats_to_select)
#     # selector = RFE(estimator, step=1, n_features_to_select=num_feats_to_select)
#     # clf = grid_search.GridSearchCV(selector, {'estimator_params': params_grid}, cv=5)
#     selector.fit(feats, labels)
#     print '...finishing rfe'
#     return selector.support_
#     # clf.best_estimator_.estimator_
#     # clf.best_estimator_.grid_scores_
#     ranking = clf.best_estimator_.ranking_
#
#     logging.info(ranking)
#     ranking = np.array(ranking)
#     ranking = np.subtract(ranking, 1)
#     return ranking

def recursive_elimination2(feats, labels, num_feats_to_select, best_rfe_param):

    estimator = svm.SVC(kernel="linear", C=best_rfe_param, random_state=1)
    selector = RFE(estimator, step=1, n_features_to_select=num_feats_to_select)
    selector = selector.fit(feats, labels)
    return selector.support_
    # return selector.ranking_

# feats, labels = get_heart_data()
# # logging.info( recursive_elimination(feats,labels))
# logging.info( recursive_elimination2(feats,labels,1))

# def recursive_elimination_top_n(feats, labels, num_feats_to_select, proportion):


def univariate_selection(feats, labels, metric):
    selector = SelectPercentile(metric, percentile=100)
    selector.fit(feats, labels)
    scores = selector.pvalues_
    return np.array(np.argsort(scores))


def get_mean_ranking_difference(first_array, second_array):
    scores = np.zeros(len(first_array))
    for i, feat_number in enumerate(first_array):
        diff = abs((np.where(second_array == feat_number)[0]) - i)[0]
        scores[i] = diff
    return np.mean(scores)


def get_ranked_indices(feats, labels, metric, num_feats_to_select=None):
    if metric == 'c':
        return univariate_selection(feats, labels, chi2)
    if metric == 'f':
        return univariate_selection(feats, labels, f_classif)
    if metric == 'r':
        return recursive_elimination(feats, labels)
    if metric == 'r2':
        return recursive_elimination2(feats, labels, num_feats_to_select)

#
# def metric_comparison(feats, labels):
#     sorted_scores_c = univariate_selection(feats, labels, chi2)
#     sorted_scores_f = univariate_selection(feats, labels, f_classif)
#     sorted_scores_r = recursive_elimination(feats, labels)
#     sorted_scores_r2 = recursive_elimination2(feats, labels, 1)
#
#     logging.info(sorted_scores_c)
#     logging.info(sorted_scores_f)
#     logging.info(sorted_scores_r)
#     logging.info(sorted_scores_r2)
#
#     logging.info(('similarity between ANOVA and chi-2 %r', get_mean_ranking_difference(sorted_scores_f, sorted_scores_c)))
#     logging.info(('similarity between ANOVA and recursive %r',
#                   get_mean_ranking_difference(sorted_scores_r, sorted_scores_f)))
#     logging.info(
#         ('similarity between chi-2 and recursive %r', get_mean_ranking_difference(sorted_scores_r, sorted_scores_c)))
#     logging.info(
#         ('similarity between recursive 1 and 2 %r', get_mean_ranking_difference(sorted_scores_r, sorted_scores_r2)))
#
#
#
#     array = np.array(range(13))
#
#     logging.info(feats[0, np.argsort(sorted_scores_r)])
#     logging.info(feats[0, np.argsort(sorted_scores_r2)])

#
# def grid_search_svc(X, Y):
#     svr = svm.SVC()
#     c_values = [.1, 1, 10]
#     gamma_values = get_gamma_from_c(c_values, X)
#     params_dict = {'C': c_values, 'gamma': gamma_values}
#     grid_search_clf = grid_search.GridSearchCV(svr, params_dict, n_jobs=4)
#     grid_search_clf.fit(X, Y)
#
#     # logging.info( grid_search_clf.grid_scores_)
#     # logging.info( grid_search_clf.score(X, Y))
#     print('best %r', grid_search_clf.best_params_)
#     return grid_search_clf.best_params_
#
#
