'''
Main function used to collect results over 10x10 folds and plot two results (line and bar) comparing three settings
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


def save_to_np_array(num_datasets,setting,n_top_feats,c_value,percent_of_priv,experiment_name):
    list_of_all_datasets = []
    for dataset_num in range(num_datasets):
        all_folds_scores = []
        for seed_num in range (num_repeats):
            output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
            n_top_feats2=''
            if setting != 'baseline':
                n_top_feats2='-{}'.format(n_top_feats)
            for inner_fold in range(num_folds):
                with open(os.path.join(output_directory,'{}-{}{}-C={}-percentpriv={}.csv'.format(setting,inner_fold,n_top_feats2,c_value,percent_of_priv)),'r') as result_file:
                    single_score = float(result_file.readline().split(',')[0])
                    all_folds_scores+=[single_score]
        list_of_all_datasets.append(all_folds_scores)
    print(np.array(list_of_all_datasets).shape)
    np.save(get_full_path('Desktop/SavedNPArrayResults/{}-{}-{}-{}-{}'.format(num_datasets,setting,n_top_feats,c_value,percent_of_priv)),list_of_all_datasets)



percent_of_priv=100
# for c_value in [1,10,100,1000]:
#     for percent_of_priv in [10,25,50,75,100]:
#         for n_top_feats in [300,500]:
#             for setting in ['lupi']:
#                 experiment_name = 'dSVM295-FIXEDC-NORMALISED-PRACTICE-10x10-tech-ALLCV-3to3-featsscaled-step0.1-top-100percentinstances'
#                 save_to_np_array(295,setting,n_top_feats,c_value,percent_of_priv,experiment_name=experiment_name)


def save_to_np_array_with_d_value(num_datasets, setting, n_top_feats, c_value, percent_of_priv, experiment_name):
    list_of_all_datasets = []
    for dataset_num in range(num_datasets):
        all_folds_scores, all_folds_d_values = [],[]
        for seed_num in range (num_repeats):
            output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
            n_top_feats2=''
            if setting != 'baseline':
                n_top_feats2='-{}'.format(n_top_feats)
            for inner_fold in range(num_folds):
                # if not (os.path.exists(os.path.join(output_directory,'{}-{}{}-{}-percentpriv={}.csv'.format(setting,inner_fold,n_top_feats2,c_value,percent_of_priv)))):
                    # print("print('--k {} --topk 300 --dataset tech --datasetnum {} --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv 100 --percentageofinstances 100 --taketopt top')".format(inner_fold,dataset_num,seed_num))
                with open(os.path.join(output_directory,'{}-{}{}-{}-percentpriv={}.csv'.format(setting,inner_fold,n_top_feats2,c_value,percent_of_priv)),'r') as result_file:
                    single_score = float(result_file.readline().split(',')[0])
                    all_folds_scores+=[single_score]
                with open(os.path.join(output_directory,'dvalue-{}{}-{}-percentpriv=100.csv'.format(inner_fold,n_top_feats2,c_value)),'r') as result_file:
                    dvalues = [float(item.strip(', []\n')) for item in result_file.readlines()]
                    print((dvalues))
                    all_folds_d_values+=[dvalues]
                    np.save(get_full_path('Desktop/SavedDvalues/{}-{}-{}-{}-{}-{}-{}'.format(setting,n_top_feats,c_value,percent_of_priv,dataset_num,seed_num,inner_fold)),dvalues)
        list_of_all_datasets.append(all_folds_scores)
    print(np.array(list_of_all_datasets).shape)
    np.save(get_full_path('Desktop/SavedNPArrayResults/{}-{}-{}-{}-{}'.format(num_datasets,setting,n_top_feats,c_value,percent_of_priv)),list_of_all_datasets)


experiment_name = 'dSVM295-SAVEd-NORMAlISED-10x10-tech-ALLCV-3to3-featsscaled-step0.1-top-100percentinstances'
save_to_np_array_with_d_value(295, 'dsvm', 300, 'cross-val', 100, experiment_name)

