__author__ = 'jt306'
import matplotlib as plt
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
experiment_name2 = '10x10-ALLCV--3to3-featsscaled-randompriv-300'


list_of_300_lupi=[]
for dataset_num in range(num_datasets):
    print ('doing dataset',dataset_num)
    all_folds_baseline, all_folds_SVM,all_folds_LUPI = [],[],[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Datla/{}/fixedCandCstar-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv={}/cross-validation{}'.format(experiment_name,dataset_num,percent_of_priv,seed_num)))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                all_folds_LUPI+=[lupi_score]
    list_of_300_lupi.append(all_folds_LUPI)


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



list_of_lupi_errors = np.array([1-mean for mean in np.mean(list_of_300_lupi,axis=1)])
list_of_lupi_errors2 = np.array([1-mean for mean in np.mean(list_of_300_lupi2,axis=1)])


print ('lupi errors',[item*100 for item in list_of_lupi_errors])
print ('lupi errors2',[item*100 for item in list_of_lupi_errors2])

lupi_error_bars = list(stats.sem(list_of_300_lupi,axis=1))
lupi_error_bars2 = list(stats.sem(list_of_300_lupi2,axis=1))




#######################################

fig = plt.figure()

plt.errorbar(list(range(num_datasets)), list_of_lupi_errors, yerr = lupi_error_bars, c='r', label='LUPI - top 300, rest privileged')
plt.errorbar(list(range(num_datasets)), list_of_lupi_errors2, yerr = lupi_error_bars, c='r', label='LUPI - random features')



fig.suptitle('Error rates{}'.format(experiment_name), fontsize=20)
plt.legend(loc='best')#bbox_to_anchor=(0.6, 1))#([line1,line2],['All features',['RFE - top 300 features']])
fig.savefig('random-normal-comparison-300.png')
plt.show()


improvements_list = []
random_improvements =0
random_worse = 0
total_improvement_over_lupi = 0
for lupi_error, random_error in zip(list_of_lupi_errors,list_of_lupi_errors2):
    total_improvement_over_lupi+=(lupi_error-random_error)
    improvements_list+=(lupi_error-random_error)
    if lupi_error>random_error:
        random_improvements+=1
    else:
        random_worse+=1
print (improvements_list)
print('lupi helped in',random_improvements,'cases vs rfe')
print('mean improvement', total_improvement_over_lupi/len(list_of_lupi_errors))


plt.bar(list(range(num_datasets),improvements_list))