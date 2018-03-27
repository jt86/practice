import os
from Get_Full_Path import get_full_path
import csv
import pandas
import numpy as np
from CollateResults import collate_single_dataset,collate_all_datasets
from ExperimentSetting import Experiment_Setting
from matplotlib import pyplot as plt
import sys

def collate_mtl_results(featsel,num_unsel_feats,weight=1):
    nn_results = np.zeros((295,10))
    for foldnum in range(10):
        filename = 'MTLresultsfile-3200units-weight{}-numfeats={}-learnrate0.0001-fold{}'.format(weight, num_unsel_feats,foldnum)
        with open(get_full_path('Desktop/Privileged_Data/MTLResults/MTL_{}_results/{}.csv'.format(featsel, filename)),'r') as results_file:
            reader = csv.reader(results_file, delimiter=',')
            for count, line in enumerate(reader):
                nn_results[int(line[1]),foldnum]=(float(line[-1]))
    assert(0 not in nn_results)
    return(nn_results)




def compare_lufe_mtl(featsel):
    # print(collate_mtl_results('rfe', 300).shape)
    mtl_results = ((collate_mtl_results(featsel.upper(), 300)))

    lufe_setting = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                            cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                            take_top_t='top', lupimethod='svmplus',
                            featsel=featsel, classifier='lufe', stepsize=0.1)
    lufe_results = collate_all_datasets(lufe_setting)


    diffs_list = (np.mean(lufe_results,axis=1)-np.mean(mtl_results,axis=1))
    print('lufe: {}, mtl: {}'.format(np.mean(lufe_results),np.mean(mtl_results)))
    print('lufe better {}; mtl better: {}; equal: {}; mean improvement={}%'.format(len(np.where(diffs_list > 0)[0]),
          len(np.where(diffs_list < 0)[0]),len(np.where(diffs_list==0)[0]),np.mean(diffs_list)*100))
    print('{} & {} & {} & {} & {:.2f}\% \\\\'.format(featsel.upper(), len(np.where(diffs_list< 0)[0]),
                            len(np.where(diffs_list==0)[0]),len(np.where(diffs_list>0)[0]),-np.mean(diffs_list)*100))




def plot_bars(mtl_results, lufe_results,featsel):
    improvements_list  = np.mean(mtl_results,axis=1)-(np.mean(lufe_results,axis=1))
    improvements_list.sort()
    name1, name2 = 'LUFe-SVM+','LUFe-MTL'
    fig = plt.figure(figsize=(15, 10))
    plt.bar(range(len(improvements_list)),improvements_list[::-1], color='black')
    # plt.title('{} VS {}\n Improvement by {} = {}%, {} of {} cases'.format(short1,short2,short1,round(np.mean(improvements_list)*100,2),len(np.where(improvements_list >= 0)[0]),len(improvements_list)))
    plt.title('{} vs {}\n Improvement by {}: mean = {}%; {} of {} cases'.format(name1,name2,name2,round(np.mean(improvements_list),2),len(np.where(improvements_list > 0)[0]),len(improvements_list)))
    plt.ylabel('Difference in accuracy score (%)\n {} better <-----> {} better'.format(name1,name2))
    plt.xlabel('dataset index (sorted by improvement)')
    plt.ylim(-20,30)
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/{}_VS_{}'.format(featsel,name1,name2)))
    plt.show()

# for featsel in ['anova','bahsic','chi2','mi','rfe']:
#     compare_lufe_mtl(featsel)
#     mtl_results = ((collate_mtl_results(featsel.upper(), 300))*100)
#     lufe_setting = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
#                             cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                             take_top_t='top', lupimethod='svmplus',
#                             featsel=featsel, classifier='lufe', stepsize=0.1)
#     lufe_results = collate_all_datasets(lufe_setting)*100
#     plot_bars(mtl_results,lufe_results,featsel)