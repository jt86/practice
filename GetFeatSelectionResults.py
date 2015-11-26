__author__ = 'jt306'


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



percent_of_priv = 100
experiment_name = '10x10-arcene-ALLCV-3to3-featsscaled-step0.25'


list_of_baselines,list_of_lupi,list_of_rfe=[],[],[]
list_of_topk = [5,10,25,50,75]

all_folds_baseline =[]
for topk in list_of_topk:
    print('doing top:',topk)
    all_folds_SVM,all_folds_LUPI = [],[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/top{}chosen/cross-validation{}'.format(experiment_name,topk,seed_num)))
        for inner_fold in range(num_folds):

            # with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
            #     baseline_score = float(baseline_file.readline().split(',')[0])
            #     all_folds_baseline+=[baseline_score]

            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,topk),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]

            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,topk),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                all_folds_LUPI+=[lupi_score]

            if topk ==5:
                print ('\n doing baseline',seed_num,inner_fold)
                with open(os.path.join(output_directory,'baseline-{}.csv').format(inner_fold),'r') as cv_baseline_file:
                    baseline_score = float(cv_baseline_file.readline().split(',')[0])
                    print (baseline_score)
                    all_folds_baseline+=[baseline_score]
                    print(all_folds_baseline)

    list_of_lupi.append(all_folds_LUPI)
    list_of_rfe.append(all_folds_SVM)
#
# list_of_lupi=np.array(list_of_lupi)
# list_of_rfe = np.array(list_of_rfe)
# print (list_of_lupi)
print ('list of lupi shape',np.array(list_of_lupi).shape)
print ('lupi',np.mean(list_of_lupi,axis=1))
print ('rfe',np.mean(list_of_rfe,axis=1))




rfe_means =np.array([1-mean for mean in np.mean(list_of_rfe,axis=1)])
lupi_means = np.array([1-mean for mean in np.mean(list_of_lupi,axis=1)])
errors = np.array(stats.sem(list_of_rfe,axis=1))
lupi_errors = np.array(stats.sem(list_of_lupi,axis=1))


baseline_list = [1-np.mean(all_folds_baseline)]*len(list_of_topk)
print(baseline_list)

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# fig.suptitle(experiment_name.title(), fontsize=20)

# print ('list of top k',list_of_topk)
# ax1.errorbar(list_of_topk, rfe_means, yerr = errors, color='b', label='SVM: trained on top features')
# ax1.errorbar(list_of_topk, lupi_means, yerr = lupi_errors, color='r', label='SVM+: lower features as privileged')
# ax1.plot(list_of_topk,baseline_list, linestyle=':', color='black',label='baseline SVM: all features')
#
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.17), fancybox=True, shadow=True, ncol=1, prop={'size': 10})
# plt.xlabel('Percentage of top-rated features used as normal information',fontsize=16)
# plt.ylabel('Accuracy score',fontsize=16)
#


improvements_list = []
improvements_count =0
worse_count = 0
total_improvement = 0
for setting1_error, setting2_error in zip(rfe_means,lupi_means):
    total_improvement+=(setting1_error-setting2_error)
    improvements_list.append(setting1_error-setting2_error)
    print ('improvements',improvements_list)
    if setting1_error>setting2_error:
        improvements_count+=1
    else:
        worse_count+=1


print (improvements_list)
print('setting2 helped in',improvements_count,'cases vs setting1')
print('mean improvement', total_improvement/len(list_of_topk))
print('len',(len(list_of_topk)))
print('improv list',improvements_list)


f, axarr = plt.subplots(2, sharex=True)

axarr[0].errorbar(list_of_topk, rfe_means, yerr = errors, color='b', label='SVM: trained on top features')
axarr[0].errorbar(list_of_topk, lupi_means, yerr = lupi_errors, color='r', label='SVM+: lower features as privileged')
axarr[0].plot(list_of_topk,baseline_list, linestyle=':', color='k',label='baseline SVM: all features')
axarr[1].bar(list_of_topk,improvements_list)

axarr[0].legend(loc='upper center', bbox_to_anchor=(0.75, 1.), fancybox=True, shadow=True, ncol=1, prop={'size': 10})

plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# axarr.xlabel('Top % features used as normal information',fontsize=16)
# axarr[0].ylabel('Accuracy score',fontsize=16)

plt.show()
