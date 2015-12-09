__author__ = 'jt306'
import matplotlib
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats

print (matplotlib.__version__)
num_repeats = 10
num_folds = 10
num_datasets=49

n_top_feats= 300
percent_of_priv = 100
experiment_name = 'f'


list_of_baselines=[]
list_of_300_rfe=[]
list_of_300_lupi=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM,all_folds_LUPI = [],[],[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        for inner_fold in range(num_folds):
            # with open(os.path.join(output_directory,'baseline.csv'),'r') as baseline_file:
            #     baseline_score = np.array([item for item in baseline_file.readline().split(',')[:-1]]).astype(np.float)
            #     all_folds_baseline+=[item for item in baseline_score]
            #
            # with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
            #     baseline_score = float(baseline_file.readline().split(',')[0])
            #     all_folds_baseline+=[baseline_score]
            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                # print(('svm score'),svm_score)
                all_folds_SVM+=[svm_score]
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                all_folds_LUPI+=[lupi_score]
        # print ('all folds svm', len(all_folds_SVM))
        # print ('all folds lupi', len(all_folds_LUPI))
    # list_of_baselines.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lupi.append(all_folds_LUPI)


experiment_name2='10x10-ALLCV-3to3-featsscaled-300'
for dataset_num in range(num_datasets):
    all_folds_baseline=[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/fixedCandCstar-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv=100/cross-validation{}'.format(experiment_name2,dataset_num,seed_num)))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]
    list_of_baselines.append(all_folds_baseline)
print('baselines',np.array(list_of_baselines).shape)

# experiment_name2 = '10x10-ALLCV-3to3-l1normalised-300'
# for dataset_num in range(num_datasets):
#     print ('doing dataset',dataset_num)
#     all_folds_baseline = []
#     for seed_num in range (10):
#         output_directory = (get_full_path('Desktop/Privileged_Data/{}/fixedCandCstar-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv=100/cross-validation{}'.format(experiment_name2,dataset_num,seed_num)))
#         for inner_fold in range(num_folds):
#             with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
#                 baseline_score = float(baseline_file.readline().split(',')[0])
#                 all_folds_baseline+=[baseline_score]
#     list_of_baselines.append(all_folds_baseline)


list_of_baseline_errors =np.array([1-mean for mean in np.mean(list_of_baselines,axis=1)])
list_of_rfe_errors = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])
list_of_lupi_errors = np.array([1-mean for mean in np.mean(list_of_300_lupi,axis=1)])

print (list_of_baseline_errors)

# print(np.argsort(list_of_baseline_errors))
#
list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_baseline_errors)]
list_of_lupi_errors = list_of_lupi_errors[np.argsort(list_of_baseline_errors)]
list_of_baseline_errors = list_of_baseline_errors[np.argsort(list_of_baseline_errors)]

# list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_lupi_errors)]
# list_of_lupi_errors = list_of_lupi_errors[np.argsort(list_of_lupi_errors)]
# list_of_baseline_errors = list_of_baseline_errors[np.argsort(list_of_lupi_errors)]

baseline_error_bars=list(stats.sem(list_of_baselines,axis=1))
rfe_error_bars = list(stats.sem(list_of_300_rfe,axis=1))
lupi_error_bars = list(stats.sem(list_of_300_lupi,axis=1))

######################################

# list_of_baselines2=[]
# list_of_300_rfe2=[]
# list_of_300_lupi2=[]
# print ('\n\n\n\n')
# for dataset_num in range(num_datasets):
#     print ('doing dataset',dataset_num)
#     all_folds_baseline, all_folds_SVM,all_folds_LUPI = [],[],[]
#     for outer_fold in range (10):
#         output_directory = (get_full_path('Desktop/Privileged_Data/TechSlice-10x4-finegrained/fixedCandCstar-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv=100/cross-validation{}'.format(dataset_num,outer_fold)))
#         for inner_fold in range(4):
#             with open(os.path.join(output_directory,'baseline.csv'),'r') as baseline_file:
#                 baseline_score = np.array([item for item in baseline_file.readline().split(',')[:-1]]).astype(np.float)
#                 all_folds_baseline+=[item for item in baseline_score]
#                 # print ('outer fold', outer_fold, 'inner fold', inner_fold, all_folds_baseline.shape)
#             with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,300),'r') as cv_svm_file:
#                 svm_score = float(cv_svm_file.readline().split(',')[0])
#                 all_folds_SVM+=[svm_score]
#             with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,300),'r') as cv_lupi_file:
#                 lupi_score = float(cv_lupi_file.readline().split(',')[0])
#                 # print (outer_fold,inner_fold,svm_score)
#                 all_folds_LUPI+=[lupi_score]
#     list_of_baselines2.append(all_folds_baseline)
#     list_of_300_rfe2.append(all_folds_SVM)
#     list_of_300_lupi2.append(all_folds_LUPI)
#
#
# list_of_baseline_errors2 = np.array([1-mean for mean in np.mean(list_of_baselines2,axis=1)])
# list_of_rfe_errors2 = np.array([1-mean for mean in np.mean(list_of_300_rfe2,axis=1)])
# list_of_lupi_errors2 = np.array([1-mean for mean in np.mean(list_of_300_lupi2,axis=1)])
#
# print ('baseline sorted',np.argsort(list_of_baseline_errors2))
# list_of_rfe_errors2 = list_of_rfe_errors2[np.argsort(list_of_baseline_errors2)]
# list_of_lupi_errors2 = list_of_lupi_errors2[np.argsort(list_of_baseline_errors2)]
# list_of_baseline_errors2 = list_of_baseline_errors2[np.argsort(list_of_baseline_errors2)]
#
# baseline_error_bars2=list(stats.sem(list_of_baselines2,axis=1))
# rfe_error_bars2 = list(stats.sem(list_of_300_rfe2,axis=1))
# lupi_error_bars2 = list(stats.sem(list_of_300_lupi2,axis=1))
# #

#######################################

fig = plt.figure()

plt.errorbar(list(range(num_datasets)), list_of_baseline_errors, yerr = baseline_error_bars, color='green', label='All features')
plt.errorbar(list(range(num_datasets)), list_of_rfe_errors, yerr = rfe_error_bars, color='blue', label='RFE - unselected features only')
plt.errorbar(list(range(num_datasets)), list_of_lupi_errors, yerr = lupi_error_bars, color='red', label='LUPI - top 300 features used as privileged')

plt.title('Reversed: using TOP 300 feats as privileged')
plt.legend(loc='best')
# plt.errorbar(list(range(num_datasets)), list_of_baseline_errors2, yerr = baseline_error_bars2, c='cyan', label='All features (original)')
# plt.errorbar(list(range(num_datasets)), list_of_rfe_errors2, yerr = rfe_error_bars2, c='k', label='RFE - top 300 features (original)')
# plt.errorbar(list(range(num_datasets)), list_of_lupi_errors2, yerr = lupi_error_bars2, c='magenta', label='LUPI - top 300, rest privileged (original)')

plt.show()


lupi_improvements =0
lupi_worse = 0
total_improvement_over_rfe, total_improvement_over_baseline, total_improvement_over_baseline2 = 0,0,0
for rfe_error, lupi_error in zip(list_of_rfe_errors,list_of_lupi_errors):
    total_improvement_over_rfe+=(rfe_error-lupi_error)
    if rfe_error>lupi_error:
        lupi_improvements+=1
    else:
        lupi_worse+=1
print('lupi helped in',lupi_improvements,'cases vs rfe')
print('mean improvement', total_improvement_over_rfe/len(list_of_rfe_errors))

lupi_improvements =0
lupi_worse = 0
for baseline_error, lupi_error in zip(list_of_baseline_errors,list_of_lupi_errors):
    total_improvement_over_baseline+=(baseline_error-lupi_error)
    if baseline_error>lupi_error:
        lupi_improvements+=1
    else:
        lupi_worse+=1
print('lupi helped in',lupi_improvements,'cases vs baseline')
print('mean improvement', total_improvement_over_baseline/len(list_of_rfe_errors))

rfe_improvements =0
rfe_worse = 0
for baseline_error, rfe_error in zip(list_of_baseline_errors,list_of_rfe_errors):
    total_improvement_over_baseline2+=(baseline_error-rfe_error)
    if baseline_error>rfe_error:
        rfe_improvements+=1
    else:
        rfe_worse+=1
print('rfe helped in',rfe_improvements,'cases vs baseline')
print('mean improvement', total_improvement_over_baseline2/len(list_of_rfe_errors))
