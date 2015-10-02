__author__ = 'jt306'

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import logging
from Get_Mean import get_mean_from, get_error_from


def get_figures(numbers_of_features_list, all_folds_SVM, all_folds_LUPI, baseline_results, output_directory, dataset, graph_directory, datasetnum):

    results = get_mean_from(all_folds_SVM)
    errors = get_error_from(all_folds_SVM)

    LUPI_results = get_mean_from(all_folds_LUPI)
    LUPI_errors = get_error_from(all_folds_LUPI)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.suptitle(dataset.title(), fontsize=20)
    # print('num feats list',len(numbers_of_features_list))
    # print('SVM results', len(results))
    # print('LUPI results',len(LUPI_results))
    # print('baseline 1 results',len(baseline_results))
    # print 'baseline 2 results',len(baseline_results2)
    # print('errors',len(errors))

    ax1.errorbar(numbers_of_features_list, results, yerr = errors, c='b', label='SVM: trained on top features')
    ax1.errorbar(numbers_of_features_list[:len(LUPI_results)], LUPI_results,   #nb number_of features list was indexed [:-1]
                 yerr = LUPI_errors, c='r', label='SVM+: lower features as privileged')

    ax1.plot(numbers_of_features_list,np.mean(baseline_results, axis=1), linestyle=':', c='black',label='baseline SVM: all features')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=1, prop={'size': 10})
    plt.xlabel('Number of top-rated features used as normal information',fontsize=16)
    plt.ylabel('Accuracy score',fontsize=16)

    plt.savefig(os.path.join(graph_directory, 'plot{}.png'.format(datasetnum)))




    #
    # differences_list = np.subtract(LUPI_results,results[:len(LUPI_results)])
    #
    #
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(211, title="Difference between SVM and SVM+ scores"+str(keyword))
    #
    #
    # ax2.bar(numbers_of_features_list[:len(LUPI_results)], differences_list[:len(LUPI_results)], 0.35)
    #
    #
    # ax1.plot(numbers_of_features_list, [0] * len(numbers_of_features_list), linestyle='-', c='black',
    #          label='baseline')
    #
    # plt.xlabel('Top n features used to train SVM')
    # plt.ylabel('Accuracy improvement with SVM+')
    #
    # plt.savefig(os.path.join(output_directory, 'differences.png'))
