import os
from Get_Full_Path import get_full_path
import csv
import pandas
import numpy as np
from CollateResults import collate_single_dataset,collate_all
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
    return(nn_results*100)

# print(collate_mtl_results('rfe',300))
# sys.exit()


def compare_lufe_mtl(featsel, lufe_setting, kernel):
    # print(collate_mtl_results('rfe', 300).shape)
    mtl_results = ((collate_mtl_results(featsel.upper(), 300)))


    lufe_results = collate_all(lufe_setting)


    diffs_list = (np.mean(lufe_results,axis=1)-np.mean(mtl_results,axis=1))
    # print('lufe: {}, mtl: {}'.format(np.mean(lufe_results),np.mean(mtl_results)))
    # print('lufe better {}; mtl better: {}; equal: {}; mean improvement={}%'.format(len(np.where(diffs_list > 0)[0]),
    #       len(np.where(diffs_list < 0)[0]),len(np.where(diffs_list==0)[0]),np.mean(diffs_list)))

    # print('{} & {} & {} & {} & {:.2f}\% \\\\'.format(featsel.upper(), len(np.where(diffs_list> 0)[0]),
    #                         len(np.where(diffs_list==0)[0]),len(np.where(diffs_list<0)[0]),-np.mean(diffs_list)))

    print('{} & {} & {} & {} & {:.2f}\% & {:.2f}\% & {:.2f}\% \\\\'.format(featsel.upper(), len(np.where(diffs_list> 0)[0]),
                            len(np.where(diffs_list==0)[0]),len(np.where(diffs_list<0)[0]), np.mean(lufe_results),
                                np.mean(mtl_results),-np.mean(diffs_list)))


def plot_bars_for_mtl(results_1, results_2, name1, name2, featsel):
    improvements_list  = np.mean(results_1, axis=1) - (np.mean(results_2, axis=1))
    improvements_list.sort()
    fig = plt.figure(figsize=(15, 10))
    plt.bar(range(len(improvements_list)),improvements_list[::-1], color='black')
    # plt.title('{} VS {}\n Improvement by {} = {}%, {} of {} cases'.format(short1,short2,short1,round(np.mean(improvements_list),2),len(np.where(improvements_list >= 0)[0]),len(improvements_list)))
    plt.title('{} vs {}\n Improvement by {}: mean = {}%; {} of {} cases'.format(name1,name2,name2,round(np.mean(improvements_list),2),len(np.where(improvements_list > 0)[0]),len(improvements_list)))
    # adding spaces to ensure y-axis arrow is centred on 0
    plt.ylabel('Difference in accuracy score (%)\n {}{} better <----------> {} better{}'.format(' '*round(1.5*(len(name2))),name1,name2,' '*round(1.5*(len(name1)))))
    plt.xlabel('dataset index (sorted by improvement)')
    plt.ylim(-25,25)

    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/{}_VS_{}.pdf'.format(featsel,name1,name2)),format='pdf')
    plt.show()


# MTL (top) vs LUFE-SVM+
# LUFe-SVM+RBF vs LUFE-SVM+
# MTL (top) v LUFe-SVM+RBF

