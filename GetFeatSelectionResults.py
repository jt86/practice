__author__ = 'jt306'


__author__ = 'jt306'

import matplotlib as plt
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats
import seaborn
num_repeats = 10
num_folds = 10



percent_of_priv = 100
experiment_name = '10x10-arcene-ALLCV0to3-featsscaled-step0.2FIRSTPARAMcv'


list_of_baselines,list_of_lupi,list_of_rfe=[],[],[]
list_of_topk = [5,10,25,50,75]

all_folds_baseline =[]
for topk in list_of_topk:
    print('doing top:',topk)
    scores_single_percentage,lupi_single_percentage = [],[]
    for seed_num in range (num_repeats):
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/top{}chosen/cross-validation{}'.format(experiment_name,topk,seed_num)))
        for inner_fold in range(num_folds):

            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,topk),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                scores_single_percentage+=[svm_score]

            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,topk),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                lupi_single_percentage+=[lupi_score]

            if topk ==75:
                with open(os.path.join(output_directory,'baseline-{}.csv').format(inner_fold),'r') as cv_baseline_file:
                    baseline_score = float(cv_baseline_file.readline().split(',')[0])
                    all_folds_baseline+=[baseline_score]


    list_of_lupi.append(lupi_single_percentage)
    list_of_rfe.append(scores_single_percentage)
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


improvements_list = []
for setting1_error, setting2_error in zip(rfe_means,lupi_means):
    improvements_list.append(setting1_error-setting2_error)

plt.subplot(2,1,1)
plt.errorbar(list_of_topk, rfe_means, yerr = errors, color='b', label='SVM: trained on top features')
plt.errorbar(list_of_topk, lupi_means, yerr = lupi_errors, color='r', label='SVM+: lower features as privileged')
plt.plot(list_of_topk,baseline_list, linestyle=':', color='k',label='baseline SVM: all features')
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 1.), fancybox=True, shadow=True, ncol=1, prop={'size': 10})
# plt.title('Error rate')
plt.ylabel('Error rate')
plt.subplot(2,1,2)
plt.bar(list_of_topk,improvements_list)
plt.ylabel('LUPI improvement over RFE')


# f, axarr = plt.subplots(2, sharex=True)
# axarr[0].errorbar(list_of_topk, rfe_means, yerr = errors, color='b', label='SVM: trained on top features')
# axarr[0].errorbar(list_of_topk, lupi_means, yerr = lupi_errors, color='r', label='SVM+: lower features as privileged')
# axarr[0].plot(list_of_topk,baseline_list, linestyle=':', color='k',label='baseline SVM: all features')
# axarr[1].bar(list_of_topk,improvements_list)
# axarr[0].legend(loc='upper center', bbox_to_anchor=(0.75, 1.), fancybox=True, shadow=True, ncol=1, prop={'size': 10})
#
#


plt.suptitle(experiment_name.title(), fontsize=20)

# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xlabel('Top % features used as normal information',fontsize=12)
# axarr[0].ylabel('Accuracy score',fontsize=16)

plt.show()
