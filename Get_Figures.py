__author__ = 'jt306'

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import logging
from Get_Mean import get_mean_from, get_error_from

def get_axis_scales(keyword):
    if 'arcene' in keyword:
        return (0,10000,0.70,1.00)
    if 'heart' in keyword:
        return (0, 14, 0.80, 0.87)
    if 'bankruptcy' in keyword:
        return (0,18,0.970,1.005)
    if 'cancer' in keyword:
        return (0,160,0.0,1.0)
    if 'ionosphere' in keyword:
        return (0,35,0.86,0.98)
    if 'vote' in keyword:
        return (0,16,0.93,0.99)
    if 'wine' in keyword:
        return (0,14,0.75,1.0)
    if 'crx' in keyword:
        return (0,45,0.65,0.95)
    if 'musk1' in keyword:
        return (0,180,0.68,0.84)
    else:
        return (0,2000,0,1)

def get_figures(numbers_of_features_list, all_folds_SVM, all_folds_LUPI, baseline_results, num_folds, output_directory, keyword):

    results = get_mean_from(all_folds_SVM)
    errors = get_error_from(all_folds_SVM)

    LUPI_results = get_mean_from(all_folds_LUPI)
    LUPI_errors = get_error_from(all_folds_LUPI)


    fig = plt.figure()
    ax1 = fig.add_subplot(111, title=" Comparison of SVM+ and SVM, for "+keyword)

    print 'num feats list',len(numbers_of_features_list)
    print 'SVM results', len(results)
    print 'LUPI results',len(LUPI_results)
    print 'baseline 1 results',len(baseline_results)
    # print 'baseline 2 results',len(baseline_results2)
    print 'errors',len(errors)w

    ax1.errorbar(numbers_of_features_list, results, #np.mean(results, axis=0),#todo changed from axis=1
                 # yerr=(np.std(results, axis=0) / np.sqrt(num_folds)),           #
                 yerr = errors,
                 c='b', label='SVM: trained on top features')

    ax1.errorbar(numbers_of_features_list[:len(LUPI_results)], LUPI_results,   #nb number_of features list was indexed [:-1]
                 # yerr=(np.std(LUPI_results, axis=0) / np.sqrt(num_folds)),
                 yerr = LUPI_errors,
                 c='r', label='SVM+: lower features as privileged')

    ax1.plot(numbers_of_features_list,np.mean(baseline_results, axis=1), linestyle=':', c='black',
             label='baseline SVM: all features')
    # #
    # ax1.plot(numbers_of_features_list,np.mean(baseline_results2, axis=1), linestyle='-.', c='green',
    #      label='baseline SVM: top t features only')

    # if 'True' in keyword:
    #     xmin, xmax,  ymin, ymax = get_axis_scales(keyword)
    #     axes = plt.gca()
    #     axes.set_xlim([xmin,xmax])
    #     axes.set_ylim([ymin,ymax])
    # else:
    #     axes = plt.gca()
    #     axes.set_ylim([0.5, 1.0])


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                      box.width, box.height * 0.8])

    # Put a legend below current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=1, prop={'size': 10})

    plt.xlabel('Top (% of t) features used as normal information (all others used as privileged)')
    plt.ylabel('Accuracy score')

    plt.savefig(os.path.join(output_directory, 'plot.png'))





    # differences_list = np.subtract(np.mean(LUPI_results, axis=1), np.mean(results,axis=1)[:len(LUPI_results)])
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
    # plt.ylabel('F-score improvement with SVM+')
    #
    # plt.savefig(os.path.join(output_directory, 'differences.png'))
