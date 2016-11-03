'''
Makes plot for experiment 3: comparing LUFe with ALL and RFE, in terms of
(a) mean improvement (b) number of datasets improved
Loads final results, which are collected and saved by Collect_Results.py
'''

__author__ = 'jt306'
import matplotlib

from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def get_plots(toporbottom,colour=''):
    folder=(get_full_path('Desktop/Privileged_Data/Collected_results/'))
    lupi_vs_rfe_list=np.load(folder+'lupi_vs_rfe_list-{}'.format(toporbottom)+'.npy')
    lupi_vs_all_list=np.load(folder+'lupi_vs_all_list-{}'.format(toporbottom)+'.npy')
    rfe_vs_all_list=np.load(folder+'rfe_vs_all_list-{}'.format(toporbottom)+'.npy')
    lupi_vs_rfe_mean=np.load(folder+'lupi_vs_rfe_mean-{}'.format(toporbottom)+'.npy')
    lupi_vs_all_mean=np.load(folder+'lupi_vs_all_mean-{}'.format(toporbottom)+'.npy')
    rfe_vs_all_mean=np.load(folder+'rfe_vs_all_mean-{}'.format(toporbottom)+'.npy')


    list_of_percentages = list(range(10,101,10))
    print(list_of_percentages)

    print(lupi_vs_rfe_list)
    fontsize=21
    matplotlib.rcParams.update({'font.size': fontsize-4})

    if colour == '':
        plt.style.use('grayscale')
    axes = plt.gca()
    axes.set_ylim([30,48])

    plt.gcf().subplots_adjust(bottom=0.1)

    plt.plot(list_of_percentages,lupi_vs_rfe_list,label='vs RFE',linestyle=':',linewidth=5, color='forestgreen')
    plt.plot(list_of_percentages,lupi_vs_all_list,label='vs ALL',linewidth=5, color='indigo')
    plt.xlabel('{} % of unselected features used'.format(toporbottom.title()),fontsize=fontsize)
    plt.ylabel('Number of datasets LUFe improves',fontsize=fontsize)
    plt.legend(loc='lower right')

    # plt.subplot.bottom = 0.60
    plt.savefig(get_full_path('Desktop/All-new-results/num-improvements-{}{}'.format(toporbottom,colour)))
    # plt.show()
    plt.close()
    axes = plt.gca()
    axes.set_ylim([0.5,4.5])
    plt.plot(list_of_percentages,lupi_vs_rfe_mean*100,label='vs RFE',linestyle=':',linewidth=5, color='forestgreen')
    plt.plot(list_of_percentages,lupi_vs_all_mean*100,label='vs ALL',linewidth=5, color='indigo')
    plt.xlabel('{} % of unselected features used'.format(toporbottom.title()),fontsize=fontsize)
    plt.ylabel('Mean improvement by LUFe (%)',fontsize=fontsize)
    plt.legend(loc='lower right',fontsize=fontsize)

    plt.savefig(get_full_path('Desktop/All-new-results/mean-improvements-{}{}'.format(toporbottom,colour)))
    # plt.show()


get_plots('bottom',colour='-colour')