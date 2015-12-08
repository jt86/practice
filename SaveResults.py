__author__ = 'jt306'

from Get_Full_Path import get_full_path
import os
import numpy as np

num_repeats = 10
num_folds = 10
num_datasets=49

n_top_feats= 300
percent_of_priv = 100
experiment_name = '10x10-ALLCV-3to3-featsscaled-300'
# experiment_name2 = '110x10-ALLCV--3to3-featsscaled-300-randompriv'


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


np.save(get_full_path('Desktop/Privileged_Data/all-results/{}-baseline'.format(experiment_name)),list_of_baselines)
np.save(get_full_path('Desktop/Privileged_Data/all-results/{}-rfe'.format(experiment_name)),list_of_300_rfe)
np.save(get_full_path('Desktop/Privileged_Data/all-results/{}-lupi'.format(experiment_name)),list_of_300_lupi)

