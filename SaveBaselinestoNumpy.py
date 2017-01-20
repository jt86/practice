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

print (sys.version)
print (matplotlib.__version__)
num_repeats = 10
num_folds = 10


method = 'RFE'
dataset='tech'


percentofinstances=100
toporbottom='top'
step=0.1


#NB if 'method' is RFE doesn't work - delete last "-{}" from line below

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
                with open(os.path.join(output_directory,'{}-{}{}.csv'.format(setting,inner_fold,n_top_feats2)),'r') as result_file:
                    single_score = float(result_file.readline().split(',')[0])
                    all_folds_scores+=[single_score]
        list_of_all_datasets.append(all_folds_scores)
    print(np.array(list_of_all_datasets).shape)
    np.save(get_full_path('Desktop/SavedNPArrayResults/{}-{}-{}-{}-{}'.format(num_datasets,setting,n_top_feats,c_value,percent_of_priv)),list_of_all_datasets)



percent_of_priv=100

###### This part saves the separate 49 and other 246 datasets as one

for n_top_feats in [300,500]:
    for setting in ['svm','baseline','lupi']:

        experiment_name = '246DATASETS-10x10-{}-ALLCV-3to3-featsscaled-step{}-100{}percentpriv-{}percentinstances-{}'.format(dataset,step,toporbottom,percentofinstances,method)
        save_to_np_array(246,setting,n_top_feats,'cross-val',100,experiment_name)

        experiment_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
        save_to_np_array(49,setting,n_top_feats,'cross-val',100,experiment_name)

        results1 = np.load(get_full_path('Desktop/SavedNPArrayResults/{}-{}-{}-{}-{}.npy'.format(49,setting,n_top_feats,'cross-val',percent_of_priv)))
        results2 = np.load(get_full_path('Desktop/SavedNPArrayResults/{}-{}-{}-{}-{}.npy'.format(246,setting,n_top_feats,'cross-val',percent_of_priv)))
        print(results1.shape,results2.shape)
        combined_results = np.vstack((results1,results2))
        np.save(get_full_path('Desktop/SavedNPArrayResults/{}-{}-{}-{}-{}'.format(295, setting, n_top_feats, 'cross-val',percent_of_priv)),combined_results)

setting='lupi'
# for n_top_feats in [300]:
