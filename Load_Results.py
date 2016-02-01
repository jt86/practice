__author__ = 'jt306'
import matplotlib
import seaborn
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats

def get_plots(top):
    folder=(get_full_path('Desktop/Privileged_Data/Collected_results/'))
    lupi_vs_rfe_list=np.load(folder+'lupi_vs_rfe_list-{}'.format(toporbottom)+'.npy')
    lupi_vs_all_list=np.load(folder+'lupi_vs_all_list-{}'.format(toporbottom)+'.npy')
    rfe_vs_all_list=np.load(folder+'rfe_vs_all_list-{}'.format(toporbottom)+'.npy')
    lupi_vs_rfe_mean=np.load(folder+'lupi_vs_rfe_mean-{}'.format(toporbottom)+'.npy')
    lupi_vs_all_mean=np.load(folder+'lupi_vs_all_mean-{}'.format(toporbottom)+'.npy')
    rfe_vs_all_mean=np.load(folder+'rfe_vs_all_mean-{}'.format(toporbottom)+'.npy')


    list_of_percentages = list(range(10,101,10))
    print(list_of_percentages)


    plt.plot(list_of_percentages,lupi_vs_rfe_list,label='vs RFE')
    plt.plot(list_of_percentages,lupi_vs_all_list,label='vs ALL')
    plt.xlabel('{} % of privileged features used'.format(toporbottom.title()))
    plt.ylabel('Number of datasets where LUFe improves')
    plt.legend(loc='lower right')
    plt.savefig(get_full_path('Desktop/All-new-results/num-improvements-{}'.format(toporbottom)))

    plt.close()

    plt.plot(list_of_percentages,lupi_vs_rfe_mean*100,label='vs RFE')
    plt.plot(list_of_percentages,lupi_vs_all_mean*100,label='vs ALL')
    plt.xlabel('{} % of privileged features used'.format(toporbottom.title()))
    plt.ylabel('Mean improvement by LUFe (%)')
    plt.legend(loc='lower right')
    plt.savefig(get_full_path('Desktop/All-new-results/mean-improvements-{}'.format(toporbottom)))



get_plots('top')