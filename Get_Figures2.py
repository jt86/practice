__author__ = 'jt306'

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import logging
from itertools import zip_longest
from scipy import stats

def get_figures(numbers_of_features_list, all_folds_SVM, all_folds_LUPI,#, all_folds_LUPI_top,# all_folds_LUPI_bottom,
                baseline_results, dataset, graph_directory, datasetnum):


    results, errors = get_mean_and_error(all_folds_SVM)
    LUPI_results, LUPI_errors = get_mean_and_error(all_folds_LUPI)
    # top_results, top_errors = get_mean_and_error(all_folds_LUPI_top)
    # bottom_results, bottom_errors = get_mean_and_error(all_folds_LUPI_bottom)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.suptitle(dataset.title(), fontsize=20)

    ax1.errorbar(numbers_of_features_list, results, yerr = errors, c='b', label='SVM: trained on top features')
    ax1.errorbar(numbers_of_features_list[:len(LUPI_results)], LUPI_results, yerr = LUPI_errors, c='r', label='SVM+: lower features as privileged')

    # ax1.errorbar(numbers_of_features_list[:len(LUPI_results)], top_results, yerr = top_errors, c='g', label='SVM+: top 50% unselected features')
    # ax1.errorbar(numbers_of_features_list[:len(LUPI_results)], bottom_results, yerr = bottom_errors, c='y', label='SVM+: bottom 50% unselected features')

    ax1.plot(numbers_of_features_list,np.mean(baseline_results, axis=1), linestyle=':', c='black',label='baseline SVM: all features')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=1, prop={'size': 10})
    plt.xlabel('Number of top-rated features used as normal information',fontsize=16)
    plt.ylabel('Accuracy score',fontsize=16)

    plt.savefig(os.path.join(graph_directory, '{}plot{}.png'.format(dataset,datasetnum)))


def get_mean_and_error(list_of_lists):
    results, errors = [], []
    izipped = list(zip_longest(*list_of_lists, fillvalue=np.nan))
    for int in range(len(izipped)):
        results.append(np.nanmean(izipped[int]))
        errors.append(stats.sem(izipped[int]))
    return results, errors

