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
n_top_feats= 300
percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1
lufecolor='forestgreen'
rfecolor='purple'
basecolor='dodgerblue'
dsvmcolor= 'red'

num_datasets=33

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
# experiment_name = 'dSVM295-FixedCpoint1-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances-1'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

count=0
dsvm_lufe=[]
for dataset_num in range(295):
    all_folds_lufe1 = []
    for seed_num in range (num_repeats):
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-FIXEDCpoint1-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe1+=[lupi_score]
    dsvm_lufe.append(all_folds_lufe1)


list_of_dsvm_errors_01 = np.array([1 - mean for mean in np.mean(dsvm_lufe, axis=1)]) * 100
dsvm_error_bars_295 = list(stats.sem(dsvm_lufe, axis=1) * 100)

print(list_of_dsvm_errors_01)

experiment_name = 'dSVM295-FixedCpoint1-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

#########################

dsvm_lufe=[]
for dataset_num in range(295):
    all_folds_lufe1 = []
    for seed_num in range (num_repeats):
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe1+=[lupi_score]
    dsvm_lufe.append(all_folds_lufe1)


list_of_dsvm_errors_unfixed = np.array([1 - mean for mean in np.mean(dsvm_lufe, axis=1)]) * 100
dsvm_error_bars_295 = list(stats.sem(dsvm_lufe, axis=1) * 100)

print(list_of_dsvm_errors_unfixed)

experiment_name = 'dSVM295-FixedCpoint1-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)



#######################








#######################

def compare_two_settings(setting_one_errors,setting_two_errors,name_one,name_two):
    improvements_list=[]
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two)
    improvements_list = np.array(improvements_list)
    print('){} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),len(setting_one_errors),np.mean(improvements_list)))
    print('mean improvement', np.mean(improvements_list))
    # with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
    #     outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
    return(improvements_list)

improvements_list = compare_two_settings(list_of_dsvm_errors_unfixed, list_of_dsvm_errors_01, 'cross-val', 'fixed')




