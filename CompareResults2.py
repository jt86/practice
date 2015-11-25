__author__ = 'jt306'
import matplotlib as plt
import seaborn
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats

num_repeats = 10
num_folds = 10
num_datasets=49

n_top_feats= 300
percent_of_priv = 100
experiment_name = '10x10-ALLCV-3to3-featsscaled-300'
experiment_name2 = '10x10-ALLCV--3to3-featsscaled-bottom50-300'


list_of_baselines,list_of_300_lupi,list_of_300_rfe=[],[],[]
for dataset_num in range(num_datasets):
    print ('doing dataset',dataset_num)
    all_folds_baseline, all_folds_SVM,all_folds_LUPI = [],[],[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/fixedCandCstar-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv={}/cross-validation{}'.format(experiment_name,dataset_num,percent_of_priv,seed_num)))
        for inner_fold in range(num_folds):

            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]

            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]
                
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                all_folds_LUPI+=[lupi_score]
    list_of_baselines.append(all_folds_baseline)
    list_of_300_lupi.append(all_folds_LUPI)
    list_of_300_rfe.append(all_folds_SVM)



percent_of_priv = 50
list_of_300_lupi2=[]
for dataset_num in range(num_datasets):
    print ('doing dataset',dataset_num)
    all_folds_baseline, all_folds_SVM,all_folds_LUPI = [],[],[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/fixedCandCstar-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv={}/cross-validation{}'.format(experiment_name2,dataset_num,percent_of_priv,seed_num)))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                all_folds_LUPI+=[lupi_score]
    list_of_300_lupi2.append(all_folds_LUPI)



list1 = list_of_300_lupi
list2 = list_of_300_lupi2

setting1 =np.array([1-mean for mean in np.mean(list1,axis=1)])
setting2 = np.array([1-mean for mean in np.mean(list2,axis=1)])

setting2 = setting2[np.argsort(setting1)]
setting1 = setting1[np.argsort(setting1)]

print ('setting2 errors',[item*100 for item in setting1])
print ('setting1 errors',[item*100 for item in setting2])

baseline_error_bars=list(stats.sem(list_of_baselines,axis=1))
lupi_error_bars = list(stats.sem(list_of_300_lupi,axis=1))
rfe_error_bars= list(stats.sem(list_of_300_rfe,axis=1))




#######################################

fig = plt.figure()

plt.errorbar(list(range(num_datasets)), setting1, yerr = baseline_error_bars, color='red', label='LUPI-300')
plt.errorbar(list(range(num_datasets)), setting2, yerr = lupi_error_bars, color='cyan', label='LUPI-300-top50%')
# plt.errorbar(list(range(num_datasets)), list_of_rfe_errors, yerr = rfe_error_bars, color='cyan', label='LUPI - 300 selected, rest priv')



fig.suptitle('Comparison - LUPI-300(all) vs LUPI-300 (bottom50%)', fontsize=20)
plt.legend(loc='best')#bbox_to_anchor=(0.6, 1))#([line1,line2],['All features',['RFE - top 300 features']])
fig.savefig('random-bottom50-comparison-300.png')
# plt.show()




improvements_list = []
improvements_count =0
worse_count = 0
total_improvement = 0
for setting1_error, setting2_error in zip(setting1,setting2):
    total_improvement+=(setting1_error-setting2_error)
    improvements_list.append(setting1_error-setting2_error)
    if setting1_error>setting2_error:
        improvements_count+=1
    else:
        worse_count+=1

print (improvements_list)
print('setting2 helped in',improvements_count,'cases vs setting1')
print('mean improvement', total_improvement/num_datasets)

print('improv list',improvements_list)
plt.bar(list(range(num_datasets)),improvements_list)
plt.show()