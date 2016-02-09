'''
Compare performance of real LUFe with using random data
'''


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

method = 'UNIVARIATE'
dataset='tech'
n_top_feats= 300
percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1

print('10x10-tech-ALLCV-3to3-featsscaled-step0.1-10bottompercentpriv-100percentinstances-{}'.format(method))
experiment_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
print (experiment_name)

random_expt_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances-RANDOM'


# keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01-{}'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv,method)
np.set_printoptions(linewidth=132)

list_of_random = []
list_of_baselines=[]
list_of_300_lupi=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_random,all_folds_LUPI = [],[],[]
    for seed_num in range (num_repeats):
        # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num)))
        random_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances-RANDOM/cross-validation{}/'.format(random_expt_name,dataset_num,n_top_feats,percentofinstances,seed_num)))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]

            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_LUPI+=[lupi_score]


            with open(os.path.join(random_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                random_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_random+=[random_score]


    list_of_baselines.append(all_folds_baseline)
    list_of_300_lupi.append(all_folds_LUPI)
    list_of_random.append(all_folds_random)

list_of_baseline_errors =np.array([1-mean for mean in np.mean(list_of_baselines,axis=1)])*100
list_of_random_errors = np.array([1-mean for mean in np.mean(list_of_random,axis=1)])*100
list_of_lupi_errors = np.array([1-mean for mean in np.mean(list_of_300_lupi,axis=1)])*100

print (list_of_baseline_errors)

list_of_random_errors = list_of_random_errors[np.argsort(list_of_baseline_errors)]
list_of_lupi_errors = list_of_lupi_errors[np.argsort(list_of_baseline_errors)]
list_of_baseline_errors = list_of_baseline_errors[np.argsort(list_of_baseline_errors)]

lupi_improvements =0
lupi_worse = 0
total_improvement_over_rfe, total_improvement_over_baseline, total_improvement_over_baseline2 = 0,0,0
improvements_list = []

for random_error, lupi_error in zip(list_of_random_errors,list_of_lupi_errors):
    total_improvement_over_rfe+=(random_error-lupi_error)
    if random_error>lupi_error:
        lupi_improvements+=1
    else:
        lupi_worse+=1
    improvements_list.append(random_error-lupi_error)

improvements_list=np.array(improvements_list)

print('lupi helped in',lupi_improvements,'cases vs standard SVM')
print('mean improvement', total_improvement_over_rfe/len(list_of_random_errors))


keyword='random-comparison'

plt.style.use('grayscale')
plt.ylabel('Error rate (%)')
# plt.subplot(2,1,2)
plt.bar(list(range(num_datasets)),improvements_list)
plt.ylabel('LUFe improvement vs LUFe-RANDOM (%)',size=18)
plt.xlabel('Dataset number',size=18)
plt.savefig(get_full_path('Desktop/All-new-results/Combined-plots/{}.png'.format(keyword)))
plt.show()

