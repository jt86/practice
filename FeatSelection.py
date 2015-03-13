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
from ParamEstimation import get_gamma_from_c

print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from Heart import get_heart_data
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import numpy as np


def recursive_elimination(feats, labels):
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(feats, labels)
    ranking = np.subtract(rfe.ranking_.reshape(len(feats[0])),1)
    return ranking

def recursive_elimination2(feats,labels,num_feats_to_select):
    c_values = [0.1, 1., 10.]
    gamma_values = get_gamma_from_c(c_values,feats)
    params_dict = {'C':c_values, 'Gamma':gamma_values}

    # print 'shape',feats.shape[1]
    if feats.shape[1]>100:
        step = 0.1
    else:
        step = 1
    # print 'num feats eliminated each time', step
    svc = SVC(kernel="linear", C=1)




    rfe = RFE(estimator=svc, n_features_to_select=num_feats_to_select, step=step)
    rfe.fit(feats, labels)
    ranking = np.subtract(rfe.ranking_.reshape(len(feats[0])),1)
    return ranking





# feats, labels = get_heart_data()
# print recursive_elimination(feats,labels)
# print recursive_elimination2(feats,labels,1)

# def recursive_elimination_top_n(feats, labels, num_feats_to_select, proportion):


def univariate_selection(feats, labels, metric):
    selector = SelectPercentile(metric, percentile=100)
    selector.fit(feats, labels)
    scores = selector.pvalues_
    return np.array(np.argsort(scores))

def get_mean_ranking_difference(first_array, second_array):
    scores = np.zeros(len(first_array))
    for i,feat_number in enumerate(first_array):
        diff = abs((np.where(second_array==feat_number)[0])-i)[0]
        scores[i]=diff
    return np.mean(scores)

def get_ranked_indices(feats, labels, metric, num_feats_to_select=None):
    if metric == 'c':
        return univariate_selection(feats,labels,chi2)
    if metric == 'f':
        return univariate_selection(feats,labels,f_classif)
    if metric == 'r':
        return recursive_elimination(feats,labels)
    if metric == 'r2':
        return recursive_elimination2(feats,labels,num_feats_to_select)


def metric_comparison(feats,labels):
    sorted_scores_c = univariate_selection(feats,labels,chi2)
    sorted_scores_f = univariate_selection(feats,labels,f_classif)
    sorted_scores_r = recursive_elimination(feats,labels)
    sorted_scores_r2 = recursive_elimination2(feats,labels,1)

    print sorted_scores_c
    print sorted_scores_f
    print sorted_scores_r
    print sorted_scores_r2


    print ('similarity between ANOVA and chi-2',get_mean_ranking_difference(sorted_scores_f,sorted_scores_c))
    print ('similarity between ANOVA and recursive',get_mean_ranking_difference(sorted_scores_r, sorted_scores_f))
    print ('similarity between chi-2 and recursive',get_mean_ranking_difference(sorted_scores_r, sorted_scores_c))
    print ('similarity between recursive 1 and 2',get_mean_ranking_difference(sorted_scores_r, sorted_scores_r2))


    # print 'first instanace',feats[0]

    array = np.array(range(13))

    # this bit is to compare to random ordering
    # print array
    # print array[:,np.argsort(sorted_scores_c)]
    # print feats[0,np.argsort(sorted_scores_c)]
    print feats[0,np.argsort(sorted_scores_r)]
    print feats[0,np.argsort(sorted_scores_r2)]




# metric_comparison(feats,labels)