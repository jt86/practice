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
n_top_feats= 300
percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1
lufecolor='forestgreen'
rfecolor='purple'
basecolor='dodgerblue'
dsvmcolor= 'red'
###########################################################
print('1')

num_datasets=295

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = 'dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

count=0
dsvm_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe1 = [], [], []
    for seed_num in range (num_repeats):
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats,seed_num))
        for inner_fold in range(num_folds):
            # if not os.path.exists(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats)):
            #     print ('datasetnum {} seednum {} inner_fold {}'.format(dataset_num,seed_num,inner_fold))
            #     count+=1
            #     print (count)
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe1+=[lupi_score]
    dsvm_lufe.append(all_folds_lufe1)


list_of_dsvm_errors = np.array([1 - mean for mean in np.mean(dsvm_lufe, axis=1)]) * 100
dsvm_error_bars_295 = list(stats.sem(dsvm_lufe, axis=1) * 100)



####################################################################### This part to get the first 40
print('2')

list_of_all=[]
list_of_300_rfe=[]
list_of_300_lufe=[]
num_datasets=49

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

list_of_300_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe = [], [], []
    for seed_num in range (num_repeats):
        # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]
            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe+=[lupi_score]

    list_of_all.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lufe.append(all_folds_lufe)


list_of_baseline_errors_49 = np.array([1 - mean for mean in np.mean(list_of_all, axis=1)]) * 100
list_of_rfe_errors_49 = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])*100
list_of_lufe_errors_49 = np.array([1 - mean for mean in np.mean(list_of_300_lufe, axis=1)]) * 100

# print (list_of_baseline_errors)
baseline_error_bars_49=list(stats.sem(list_of_all, axis=1) * 100)
rfe_error_bars_49 = list(stats.sem(list_of_300_rfe,axis=1)*100)
lufe_error_bars_49 = list(stats.sem(list_of_300_lufe, axis=1) * 100)

######################################################################## This part to get the next 246
print('3')
num_datasets=246

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = '246DATASETS-10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

list_of_all=[]
list_of_300_rfe=[]
list_of_300_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe = [], [], []
    for seed_num in range (num_repeats):
        # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]
            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe+=[lupi_score]

    list_of_all.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lufe.append(all_folds_lufe)


num_datasets=190

list_of_baseline_errors_246 =(np.array([1-mean for mean in np.mean(list_of_all,axis=1)])*100)
list_of_rfe_errors_246 = (np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])*100)
list_of_lufe_errors_246 = (np.array([1 - mean for mean in np.mean(list_of_300_lufe, axis=1)]) * 100)

list_of_rfe_errors = np.hstack((list_of_rfe_errors_49,list_of_rfe_errors_246))[:num_datasets]
list_of_lufe_errors = np.hstack((list_of_lufe_errors_49,list_of_lufe_errors_246))[:num_datasets]
list_of_all_errors = np.hstack((list_of_baseline_errors_49, list_of_baseline_errors_246))[:num_datasets]



def compare_two_settings(setting_one_errors,setting_two_errors,name_one,name_two):
    improvements_list=[]
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two)
    improvements_list = np.array(improvements_list)
    print('{} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),len(setting_one_errors),np.mean(improvements_list)))
    # with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
    #     outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
    return(improvements_list)




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

def compare_two_settings(setting_one_errors,setting_two_errors,name_one,name_two):
    improvements_list=[]
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two) # this value is positive if error one > error two, ie error two improves
    improvements_list = np.array(improvements_list)
    print('{} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),len(setting_one_errors),np.mean(improvements_list)))
    # with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
    #     outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
    return(improvements_list)

compare_two_settings(list_of_rfe_errors,list_of_dsvm_errors_fixed1000,'rfe','fixed_dsvm 1000')
compare_two_settings(list_of_rfe_errors,list_of_dsvm_errors_fixed100,'rfe','fixed_dsvm 100')
compare_two_settings(list_of_rfe_errors,list_of_dsvm_errors_fixed10,'rfe','fixed_dsvm 10')
compare_two_settings(list_of_rfe_errors,list_of_dsvm_errors_fixed1,'rfe','fixed_dsvm 1')

compare_two_settings(list_of_lufe_errors,list_of_dsvm_errors_fixed1000,'rfe','fixed_dsvm 1000')
compare_two_settings(list_of_lufe_errors,list_of_dsvm_errors_fixed100,'rfe','fixed_dsvm 100')
compare_two_settings(list_of_lufe_errors,list_of_dsvm_errors_fixed10,'rfe','fixed_dsvm 10')
compare_two_settings(list_of_lufe_errors,list_of_dsvm_errors_fixed1,'rfe','fixed_dsvm 1')