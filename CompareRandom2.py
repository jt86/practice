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
num_datasets=49

method = 'RFE'
dataset='tech'
n_top_feats= 300
percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
random_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-RANDOM'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
print(experiment_name)
print('10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances')
if experiment_name == ('10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'):
    print (True)
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)


list_of_300_rfe=[]
list_of_300_random=[]
for dataset_num in range(num_datasets):
    all_folds_rfe,all_folds_random = [],[]
    for seed_num in range (num_repeats):
        # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        true_rfe_dir = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
        random_rfe_dir = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances-RANDOM/cross-validation{}/'.format(random_name,dataset_num,n_top_feats,percentofinstances,seed_num))

        for inner_fold in range(num_folds):
            with open(os.path.join(true_rfe_dir,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_rfe+=[lupi_score]
        for inner_fold in range(num_folds):
            with open(os.path.join(random_rfe_dir,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_random_file:
                lupi_score = float(cv_random_file.readline().split(',')[0])
                all_folds_random+=[lupi_score]

    list_of_300_rfe.append(all_folds_rfe)
    list_of_300_random.append(all_folds_random)

list_of_rfe_errors = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])*100
list_of_random_errors = np.array([1-mean for mean in np.mean(list_of_300_random,axis=1)])*100

# print (list_of_baseline_errors)

list_of_random_errors = list_of_random_errors[np.argsort(list_of_rfe_errors)]
list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_rfe_errors)]



rfe_error_bars = list(stats.sem(list_of_300_rfe,axis=1)*100)
lupi_error_bars = list(stats.sem(list_of_300_random,axis=1)*100)


if method=='UNIVARIATE':
    method='ANOVA'


plt.style.use('grayscale')
# plt.subplot(2,1,1)

ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.errorbar(list(range(num_datasets)), list_of_random_errors, yerr = lupi_error_bars, label='LUFe-{}-{}'.format(method,n_top_feats),marker='x',markersize=3,fillstyle='none')#, linestyle='-.')
ax0.errorbar(list(range(num_datasets)), list_of_rfe_errors, yerr = rfe_error_bars, label='{}-{}'.format(method,n_top_feats),marker='s',markersize=3,fillstyle='none')#, linestyle='-.')
# ax0.errorbar(list(range(num_datasets)), list_of_baseline_errors, yerr = baseline_error_bars, label='ALL',linestyle='--')
ax0.legend(loc='lower right',prop={'size':10})
ax0.set_ylabel('Error rate (%)')




# plt.savefig(get_full_path('Desktop/All-new-results/{}.png'.format(keyword)))
outputfile=open('/Volumes/LocalDataHD/j/jt/jt306/Desktop/All-new-results2/a','a')
# plt.show()

real_is_better_count =0
random_is_better_count = 0
total_improvement_over_rfe, total_improvement_over_baseline, total_improvement_over_baseline2 = 0,0,0
improvements_list = []

for rfe_error, random_error in zip(list_of_rfe_errors,list_of_random_errors):
    total_improvement_over_rfe+=(rfe_error-random_error)
    if rfe_error<random_error:
        real_is_better_count+=1
    if rfe_error==random_error:
        print('same error rate!')
    else:
        random_is_better_count+=1
    improvements_list.append(rfe_error-random_error)

improvements_list=np.array(improvements_list)
print(improvements_list)

print('real helped in',real_is_better_count,'cases vs  random')
print('mean improvement', total_improvement_over_rfe/len(list_of_rfe_errors))
outputfile.write('\nlupi helped in {} cases vs standard SVM'.format(real_is_better_count))
outputfile.write('\nmean improvement={}'.format(total_improvement_over_rfe/len(list_of_rfe_errors)))

################################################
#
# lupi_improvements =0
# lupi_worse = 0
# for baseline_error, lupi_error in zip(list_of_baseline_errors,list_of_random_errors):
#     total_improvement_over_baseline+=(baseline_error-lupi_error)
#     if baseline_error>lupi_error:
#         lupi_improvements+=1
#     else:
#         lupi_worse+=1
# print('lupi helped in',lupi_improvements,'cases vs all-feats-baseline')
# print('mean improvement', total_improvement_over_baseline/len(list_of_rfe_errors))
#
#
# a.write('\nlupi helped in {} cases vs all-feats-baseline'.format(lupi_improvements))
# outputfile.write('\nmean improvement={}'.format(total_improvement_over_baseline/len(list_of_rfe_errors)))
#
# #############################################
#
# rfe_improvements =0
# rfe_worse = 0
# for baseline_error, rfe_error in zip(list_of_baseline_errors,list_of_rfe_errors):
#     total_improvement_over_baseline2+=(baseline_error-rfe_error)
#     if baseline_error>rfe_error:
#         rfe_improvements+=1
#     else:
#         rfe_worse+=1
# print('feat selection helped in',rfe_improvements,'cases vs all-feats-baseline')
# print('mean improvement', total_improvement_over_baseline2/len(list_of_rfe_errors))
# outputfile.write('\nrfe helped in {} cases vs baseline'.format(rfe_improvements))
# outputfile.write('\nmean improvement={}'.format(total_improvement_over_baseline2/len(list_of_rfe_errors)))
# outputfile.close()
##########################################

ax1 = plt.subplot2grid((3,1), (2,0))
# ax1.plot(y, x)


# ax1.subplot(2,1,2)
ax1.bar(list(range(num_datasets)),improvements_list)
ax1.set_ylabel('Improvement by LUFe(%)')
ax1.set_ylim(-4,10)
ax1.set_xlabel('Dataset number')
# plt.axes('')
plt.savefig(get_full_path('/Volumes/LocalDataHD/j/jt/jt306/Desktop/All-new-results2/Combined-plots/{}.png'.format(keyword)))
plt.show()


list_of_worse = np.where(improvements_list<0)
print(list_of_worse)

total_lupi = np.sum(list_of_random_errors)/49
print ('lupi',100-total_lupi)

# total_all = np.sum(list_of_baseline_errors)/49
# print ('all',100-total_all)

total_rfe = np.sum(list_of_rfe_errors)/49
print ('rfe',100-total_rfe)

print (total_rfe-total_lupi)
# print (total_all-total_lupi)
# print (total_all-total_rfe)