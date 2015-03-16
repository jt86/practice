__author__ = 'jt306'

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_figures(numbers_of_features_list, results, LUPI_results, baseline_results, num_folds, output_directory, keyword,bottom_n_percent):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, title=" Comparison of SVM+ and basic SVM, for "+keyword)

    ax1.errorbar(numbers_of_features_list, np.mean(results, axis=1),
                 yerr=(np.std(results, axis=1) / np.sqrt(num_folds)),
                 c='b', label='SVM: trained on top features')
    print "number of feats",len(numbers_of_features_list),"number of results", len(np.mean(LUPI_results, axis=1))


    if bottom_n_percent==0:
        ax1.errorbar(numbers_of_features_list[:-1], np.mean(LUPI_results, axis=1),
                     yerr=(np.std(LUPI_results, axis=1) / np.sqrt(num_folds)),
                     c='r', label='SVM+: lower features as privileged')
    else:
        ax1.errorbar(numbers_of_features_list, np.mean(LUPI_results, axis=1),
                     yerr=(np.std(LUPI_results, axis=1) / np.sqrt(num_folds)),
                     c='r', label='SVM+: lower features as privileged')


    ax1.plot(numbers_of_features_list,np.mean(baseline_results, axis=1), linestyle='-', c='black',
             label='baseline SVM: all features')



    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                      box.width, box.height * 0.8])

    # Put a legend below current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=1, prop={'size': 10})

    plt.xlabel('Top n features used to train SVM')
    plt.ylabel('F-score')

    plt.savefig(os.path.join(output_directory, 'plot.png'))

    all_but_one_results = np.mean(results,axis=1)[:-1]
    # differences_list = np.subtract(np.mean(LUPI_results, axis=1), np.mean(results, axis=1))

    if bottom_n_percent ==0:
        differences_list = np.subtract(np.mean(LUPI_results, axis=1), all_but_one_results)
    else:
        differences_list = np.subtract(np.mean(LUPI_results, axis=1), np.mean(results,axis=1))

    width = 0.35  # the width of the bars

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(211, title="Difference between SVM and SVM+ scores"+str(keyword))

    if bottom_n_percent == 0:
        ax2.bar(numbers_of_features_list[:-1], differences_list, width)
    else:
        ax2.bar(numbers_of_features_list, differences_list, width)


    ax1.plot(numbers_of_features_list, [0] * len(numbers_of_features_list), linestyle='-', c='black',
             label='baseline')

    plt.xlabel('Top n features used to train SVM')
    plt.ylabel('F-score improvement with SVM+')

    plt.savefig(os.path.join(output_directory, 'differences.png'))
    # print differences_list