__author__ = 'jt306'

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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


    xmin, xmax,  ymin, ymax = get_axis_scales(keyword)
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])

    print xmin, xmax,  ymin, ymax

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