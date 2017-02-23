'''
Main function used to collect results over 10x10 folds for experiments involving subsets of priv data (10,25,50%)
'''

__author__ = 'jt306'
import matplotlib
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats
import matplotlib.cm as cm
import csv

num_repeats = 10
num_folds = 10
method = 'RFE'
dataset='tech'
percentofinstances=100
toporbottom='top'
step=0.1

# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'

np.set_printoptions(linewidth=132)

experiment = 'LUFeSubset-10x10-tech-ALLCV-3to3-featsscaled-step01-100percentinstances'#/{}{}/top{}chosen-{}percentinstances/')#.format(dataset,cmin,cmax,stepsize,percentageofinstances,dataset,datasetnum,topk,percentageofinstances))



def save_to_np_array(num_datasets,setting,n_top_feats,c_value,percent_of_priv,experiment_name):
    list_of_all_datasets = []
    for dataset_num in range(num_datasets):
        all_folds_scores = []
        for seed_num in range (num_repeats):

            # Change Subsets Incomplete to PRivileged_Data

            output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Subsets_Incomplete/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/top-{}'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num,percent_of_priv))
            # n_top_feats2=''
            # if setting != 'baseline':
            #     n_top_feats2='-{}'.format(n_top_feats)
            for inner_fold in range(num_folds):
                with open(os.path.join(output_directory,'{}-{}-{}.csv'.format(setting,inner_fold,n_top_feats)),'r') as result_file:
                    single_score = float(result_file.readline().split(',')[0])
                    all_folds_scores+=[single_score]
        list_of_all_datasets.append(all_folds_scores)
    print(np.array(list_of_all_datasets).shape)
    np.save(get_full_path('Desktop/SavedNPArrayResults/{}/{}-{}-{}-{}-{}'.format(dataset,num_datasets,setting,n_top_feats,c_value,percent_of_priv)),list_of_all_datasets)



percent_of_priv=100
for c_value in ['cross-val']:
    for percent_of_priv in [10,25,50]:
        for n_top_feats in [300]:
            for setting in ['lupi']:
                # experiment_name = 'dSVM295-FIXEDC-NORMALISED-PRACTICE-10x10-tech-ALLCV-3to3-featsscaled-step0.1-top-100percentinstances'
                experiment_name = 'LUFeSubset-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100percentinstances'
                save_to_np_array(1,setting,n_top_feats,c_value,percent_of_priv,experiment_name=experiment_name)
