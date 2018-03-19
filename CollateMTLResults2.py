import os
from Get_Full_Path import get_full_path
import csv
import pandas
import numpy as np
from CollateResults import collate_single_dataset,collate_all_datasets
from ExperimentSetting import Experiment_Setting
from matplotlib import pyplot as plt
import sys

def collate_mtl_results(featsel,num_unsel_feats,weight=1.0):
    nn_results = np.zeros((295,10))
    for foldnum in range(10):
        filename = 'MTLresultsfile-3200units-weight{}-numfeats={}-learnrate0.0001-fold{}'.format(weight, num_unsel_feats,foldnum)
        with open(get_full_path('Desktop/Privileged_Data/MTLResults/MTL_{}_results/{}.csv'.format(featsel, filename)),'r') as results_file:
            reader = csv.reader(results_file, delimiter=',')
            for count, line in enumerate(reader):
                nn_results[int(line[1]),foldnum]=(float(line[-1]))

    # assert(0 not in nn_results)
    if 0 in nn_results:
        # print(nn_results)
        print(np.argwhere(nn_results==0))
    return(nn_results)


all=[]
#
# for num_unsel_feats in [1,2,3]+[item for item in range(10,300,10)]+[item for item in range (1000,2200,100)]+['all']:
#     print('\n {} \n'.format(num_unsel_feats))
#     all.append(np.mean(collate_mtl_results('ANOVA',num_unsel_feats)))


num_unsel_feats = 300
print('\n {} \n'.format(num_unsel_feats))
print((collate_mtl_results('RFE',num_unsel_feats)))

# plt.plot([item for item in [1,2,3]+[item for item in range(10,300,10)]+[item for item in range (1000,2200,100)]],22000)
# ax = plt.subplot()
# ax.set_xscale("log")
# plt.show()

# find the 0 scores in BAHSIC and CHI2. Re run. Also rerun the ALL unselectetd fetrus setting