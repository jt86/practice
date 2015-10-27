__author__ = 'jt306'
import matplotlib as plt
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
list_of_values = [300,500]
from scipy import stats

x = list(range(49))
y = list(range(49))


list_of_baselines=[]
list_of_300_rfe=[]
list_of_300_lupi=[]

for dataset_num in range(49):
    print ('doing dataset',dataset_num)
    all_folds_baseline=[]
    all_folds_SVM,all_folds_LUPI = [],[]
    for outer_fold in range (10):
        output_directory = (get_full_path('Desktop/Privileged_Data/10x10tech/ALL/fixedC-10fold-tech-{}-RFE-baseline-step=0.1-percent_of_priv=100/cross-validation{}'.format(dataset_num,outer_fold)))

        with open(os.path.join(output_directory,'baseline.csv'),'r') as baseline_file:
            baseline_i_list = baseline_file.readline().split(',')[:-1]
            baseline_i_list = list(map(float, baseline_i_list))
            # print (baseline_i_list)
            all_folds_baseline+=baseline_i_list
            # print (len(list_of_baselines))


        for inner_fold in range(10):
            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,300),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]

            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,300),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                # print (outer_fold,inner_fold,svm_score)
                all_folds_LUPI+=[lupi_score]


    list_of_baselines.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lupi.append(all_folds_LUPI)




list_of_baseline_errors = np.array([1-mean for mean in np.mean(list_of_baselines,axis=1)])
list_of_rfe_errors = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])
list_of_lupi_errors = np.array([1-mean for mean in np.mean(list_of_300_lupi,axis=1)])



    # , rfe_error_bars, lupi_error_bars = [],[],[]


print ('baseline sorted',np.argsort(list_of_baseline_errors))
list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_baseline_errors)]
list_of_lupi_errors = list_of_lupi_errors[np.argsort(list_of_baseline_errors)]
list_of_baseline_errors = list_of_baseline_errors[np.argsort(list_of_baseline_errors)]

baseline_error_bars=list(stats.sem(list_of_baselines,axis=1))
rfe_error_bars = list(stats.sem(list_of_300_rfe,axis=1))
lupi_error_bars = list(stats.sem(list_of_300_lupi,axis=1))
#
fig = plt.figure()
# plt.plot(list(range(49)),list_of_baseline_errors,c='g',label='All features', yerr=baseline_error_bars)
plt.errorbar(list(range(49)), list_of_baseline_errors, yerr = baseline_error_bars, c='g', label='All features')
plt.errorbar(list(range(49)), list_of_rfe_errors, yerr = rfe_error_bars, c='b', label='RFE - top 300 features')
plt.errorbar(list(range(49)), list_of_lupi_errors, yerr = lupi_error_bars, c='r', label='LUPI - top 300, rest privileged')


# plt.plot(list(range(49)),list_of_baseline_errors,c='g',label='All features', yerr=baseline_error_bars)
# plt.plot(list(range(49)),list_of_rfe_errors,c='b',label='RFE - top 300 features')
# plt.plot(list(range(49)),list_of_lupi_errors,c='r',label='LUPI - top 300, rest privileged')






fig.suptitle('TechTC-300 - Error rates', fontsize=20)
plt.legend()#([line1,line2],['All features',['RFE - top 300 features']])
fig.savefig('newplot')
plt.show()


# lupi_improvements =0
# lupi_worse = 0
# for rfe_error, lupi_error in zip(list_of_rfe_errors,list_of_lupi_errors):
#     if rfe_error>lupi_error:
#         lupi_improvements+=1
#     else:
#         lupi_worse+=1
# print('lupi helped in',lupi_improvements,'cases vs rfe')
# print('lupi worsened in',lupi_worse,'cases vs rfe')
#
# lupi_improvements =0
# lupi_worse = 0
# for baseline_error, lupi_error in zip(list_of_baseline_errors,list_of_lupi_errors):
#     if baseline_error>lupi_error:
#         lupi_improvements+=1
#     else:
#         lupi_worse+=1
# print('lupi helped in',lupi_improvements,'cases vs baseline')
# print('lupi worsened in',lupi_worse,'cases vs baseline')
#
#
# rfe_improvements =0
# rfe_worse = 0
# for baseline_error, rfe_error in zip(list_of_baseline_errors,list_of_rfe_errors):
#     if baseline_error>rfe_error:
#         rfe_improvements+=1
#     else:
#         rfe_worse+=1
# print('rfe helped in',rfe_improvements,'cases vs baseline')
# print('rfe worsened in',rfe_worse,'cases vs baseline')
#