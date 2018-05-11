# from CollateResults import
from Get_Full_Path import get_full_path
from ExperimentSetting import Experiment_Setting
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mutual_info_score
from SingleFoldSlice import get_train_and_test_this_fold, get_norm_priv
import seaborn
from ExperimentSetting import Experiment_Setting
from CollateResults import compare_two_settings, get_graph_labels

def get_settings(featsel):
    sb = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                            take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
    s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                            take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                            take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')

    return(sb,s1,s2)

def all_plots(sb,s1,s2,featsel,chapname):

    pair = [(s1, s2), (sb, s2), (sb, s1)]

    fig, ax = plt.subplots(figsize=(10, 15))
    plt.subplots_adjust(hspace=0.3)
    for num, (s1, s2) in enumerate(pair):
        improvements_list = compare_two_settings(s1, s2)
        improvements_list.sort()
        name1, name2 = get_graph_labels(s1).split('-')[0], get_graph_labels(s2).split('-')[0]
        plt.subplot(3, 1, num+1)
        plt.bar(range(len(improvements_list)), improvements_list[::-1], color='black')
        num_imprs =  len(np.where(improvements_list > 0)[0])/len(improvements_list)*100
        plt.title('{} vs {}:  {}% higher accuracy (improved in {:.1f}% of cases)'.format(name2, name1,
                round(np.mean(improvements_list),2),num_imprs))
        plt.ylabel('Difference in accuracy score (%)\n {} better <-----> {} better'.format(name1, name2))
        plt.xlabel('dataset index (sorted by improvement)')
        plt.ylim(-20, 30)
        plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/allplots-{}'.format(chapname,featsel)))
    plt.show()

# for featsel in ['rfe','anova','bahsic','chi2','mi','rfe']:
#     settings = get_settings(featsel)
#     all_plots(featsel)