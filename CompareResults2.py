__author__ = 'jt306'
import matplotlib as plt
import seaborn
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats

num_datasets=49


#load results for baseline, rfe, lupi, as 10x10 arrays, saved using SaveResults.py
experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1      '
list_of_baselines= np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-baseline.npy'.format(experiment_name)))
list_of_300_rfe = np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-rfe.npy'.format(experiment_name)))
list_of_300_lupi = np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-lupi.npy'.format(experiment_name)))


list1 = list_of_300_rfe
list2 = list_of_300_lupi

setting1 =np.array([1-mean for mean in np.mean(list1,axis=1)])
setting2 = np.array([1-mean for mean in np.mean(list2,axis=1)])
# setting2 = setting2[np.argsort(setting1)]
# setting1 = setting1[np.argsort(setting1)]

print ('setting2 errors',[item*100 for item in setting1])
print ('setting1 errors',[item*100 for item in setting2])


#######################################

#plot comparison between two lists setting1 and setting2, sorted by setting1

baseline_error_bars=list(stats.sem(list_of_baselines,axis=1))
lupi_error_bars = list(stats.sem(list_of_300_lupi,axis=1))
rfe_error_bars= list(stats.sem(list_of_300_rfe,axis=1))
fig = plt.figure()
plt.errorbar(list(range(num_datasets)), setting1, yerr = baseline_error_bars, color='green', label='all features')
plt.errorbar(list(range(num_datasets)), setting2, yerr = lupi_error_bars, color='blue', label='RFE-300')
# plt.errorbar(list(range(num_datasets)), list_of_rfe_errors, yerr = rfe_error_bars, color='cyan', label='LUPI - 300 selected, rest priv')


# fig.suptitle('Comparison - all features vs RFE-300', fontsize=20)
# plt.legend(loc='best')#bbox_to_anchor=(0.6, 1))#([line1,line2],['All features',['RFE - top 300 features']])
# fig.savefig('random-bottom50-comparison-300.png')
# # plt.show()


#plot corresponding bar graphs showing relative improvement of setting2 over setting1


improvements_list = []
total_improvement = 0
for setting1_error, setting2_error in zip(setting1,setting2):
    total_improvement+=(setting1_error-setting2_error)
    improvements_list.append(setting1_error-setting2_error)
    # if setting1_error>setting2_error:
    #     improvements_count+=1
    # else:
    #     worse_count+=1

improvements_list=np.array(improvements_list)

improvements_count=(np.array(improvements_list)>0).sum()
print('setting2 helped in',improvements_count,'cases vs setting 1')
print('setting2 didnt help in',49-improvements_count,'cases vs setting 1')
print('mean improvement', total_improvement/num_datasets)

print('unsorted improvements list',improvements_list)
plt.bar(list(range(num_datasets)),improvements_list)
# plt.show()

print('sorted')
list49 = np.array((list(range(49))))
print (list49[np.argsort(improvements_list)])

print (sorted(improvements_list))
print(np.max(improvements_list))
print(improvements_list[10])
print(improvements_list[36])
print (np)

list_of_worse = np.where(improvements_list<-0.01)
list_of_better = np.where(improvements_list>0.05)
print(list_of_worse)
print(list_of_better)
