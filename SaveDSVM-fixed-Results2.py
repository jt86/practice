
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
dsvm_lufe=[]



num_datasets=190

dsvm_lufe_1,dsvm_lufe_10,dsvm_lufe_100,dsvm_lufe_1000=[],[],[],[]
for dataset_num in range(190):
    all_folds_lufe1, all_folds_lufe10, all_folds_lufe100, all_folds_lufe1000 = [],[],[],[]
    for seed_num in range (num_repeats):
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-FIXEDC-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats, seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'lupi-{}-{}-C=1.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                all_folds_lufe1+=[float(cv_lupi_file.readline().split(',')[0])]
            with open(os.path.join(output_directory,'lupi-{}-{}-C=10.csv').format(inner_fold, n_top_feats),'r') as cv_lupi_file:
                all_folds_lufe10 += [float(cv_lupi_file.readline().split(',')[0])]
            with open(os.path.join(output_directory,'lupi-{}-{}-C=100.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                all_folds_lufe100+=[float(cv_lupi_file.readline().split(',')[0])]
            with open(os.path.join(output_directory,'lupi-{}-{}-C=1000.csv').format(inner_fold, n_top_feats),'r') as cv_lupi_file:
                all_folds_lufe1000 += [float(cv_lupi_file.readline().split(',')[0])]



    dsvm_lufe_1.append(all_folds_lufe1)
    dsvm_lufe_10.append(all_folds_lufe10)
    dsvm_lufe_100.append(all_folds_lufe100)
    dsvm_lufe_1000.append(all_folds_lufe1000)


list_of_dsvm_errors_fixed1 = np.array([1 - mean for mean in np.mean(dsvm_lufe_1, axis=1)]) * 100
print(list_of_dsvm_errors_fixed1)

list_of_dsvm_errors_fixed10 = np.array([1 - mean for mean in np.mean(dsvm_lufe_10, axis=1)]) * 100
print(list_of_dsvm_errors_fixed10)

list_of_dsvm_errors_fixed100 = np.array([1 - mean for mean in np.mean(dsvm_lufe_100, axis=1)]) * 100
print(list_of_dsvm_errors_fixed100)

list_of_dsvm_errors_fixed1000 = np.array([1 - mean for mean in np.mean(dsvm_lufe_1000, axis=1)]) * 100
print(list_of_dsvm_errors_fixed1000)





##############
count=0
dsvm_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_lufe1 = []
    for seed_num in range (num_repeats):
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-FIXEDCpoint1-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe1+=[lupi_score]
    dsvm_lufe.append(all_folds_lufe1)


list_of_dsvm_errors_fixed01 = np.array([1 - mean for mean in np.mean(dsvm_lufe, axis=1)]) * 100
dsvm_error_bars_295 = list(stats.sem(dsvm_lufe, axis=1) * 100)

print(list_of_dsvm_errors_fixed01)

experiment_name = 'dSVM295-FixedCpoint1-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

#########################

dsvm_lufe=[]
for dataset_num in range(num_datasets):
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

# plt.plot(range(num_datasets),list_of_dsvm_errors_unfixed,label='not fixed')
# plt.plot(range(num_datasets),list_of_dsvm_errors_fixed01,label='0.1')
# plt.plot(range(num_datasets),list_of_dsvm_errors_fixed1,label='1.')
# plt.plot(range(num_datasets),list_of_dsvm_errors_fixed10,label='10.')
# plt.plot(range(num_datasets),list_of_dsvm_errors_fixed100,label='100.')
# plt.plot(range(num_datasets),list_of_dsvm_errors_fixed1000,label='1000.')
# plt.legend(loc='best')
# plt.show()


def compare_two_settings(setting_one_errors,setting_two_errors,name_one,name_two):
    improvements_list=[]
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two) # this value is positive if error one > error two, ie error two improves
    improvements_list = np.array(improvements_list)
    print('{} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),len(setting_one_errors),np.mean(improvements_list)))
    # with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
    #     outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
    return(improvements_list)

improvements_list = compare_two_settings(list_of_dsvm_errors_unfixed, list_of_dsvm_errors_fixed1000, 'cross-val', 'fixed')

all_settings = [(list_of_dsvm_errors_unfixed,'unfixed'),(list_of_dsvm_errors_fixed01,'fixed0.1'),(list_of_dsvm_errors_fixed1,'fixed1'),(list_of_dsvm_errors_fixed10,'fixed10'),(list_of_dsvm_errors_fixed100,'fixed100'),(list_of_dsvm_errors_fixed1000,'fixed1000')]
for item in all_settings:
    for item2 in all_settings:
        compare_two_settings(item[0],item2[0],item[1],item2[1])

    print('\n')