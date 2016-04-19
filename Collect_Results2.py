C__author__ = 'jt306'
# import matplotlib
from Get_Full_Path import get_full_path
import os
# from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats

__author__ = 'jt306'
# import matplotlib
from Get_Full_Path import get_full_path
import os
# from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats

# print (matplotlib.__version__)
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

def compare_two_results(list1,list2,name1,name2):
    list1_better, list2_better = 0,0
    for error1, error2 in zip (list1,list2):
        if error1 > error2:
            list2_better += 1
        if error1 < error2:
            list1_better += 1
    print ('{} helped in {} cases vs {}'.format(name1,list1_better,name2))
    print ('{} was worse in {} cases vs {}'.format(name1,list2_better,name2))

    mean = (np.mean(list1)-np.mean(list2))
    return (mean,list1_better)


all_settings_list=[]
for percent_of_priv in list(range(100,101,10)):
    print('\n\n\n\n percent of priv',percent_of_priv)
    experiment_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-RFE'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
    experiment_name2 = 'CombinedNormalPriv-10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances'.\
        format(dataset, step, percent_of_priv,toporbottom,percentofinstances,method)
    # print (experiment_name)

    keyword = '{}-{}feats-{}-3to3-{}instances-{}{}priv-step0.1-NEW'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
    np.set_printoptions(linewidth=132)
    list_of_baselines=[]
    list_of_300_rfe=[]
    list_of_300_lupi=[]
    list_of_combined = []
    for dataset_num in range(num_datasets):
        all_folds_baseline, all_folds_SVM,all_folds_LUPI,all_folds_combined = [],[],[],[]
        for seed_num in range (num_repeats):
            # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
            output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num)))
            output_directory2= (get_full_path(
                'Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(
                    experiment_name2, dataset_num, n_top_feats, percentofinstances, seed_num)))

            for inner_fold in range(num_folds):
                with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                    baseline_score = float(baseline_file.readline().split(',')[0])
                    all_folds_baseline+=[baseline_score]
                with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                    svm_score = float(cv_svm_file.readline().split(',')[0])
                    all_folds_SVM+=[svm_score]
                with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                    lupi_score = float(cv_lupi_file.readline().split(',')[0])
                    all_folds_LUPI+=[lupi_score]
                with open(os.path.join(output_directory2, 'combined_score-{}-{}.csv').format(inner_fold, n_top_feats),'r') as cv_combined_file:
                    combined_score = float(cv_combined_file.readline().split(',')[0])
                    all_folds_combined += [combined_score]


        list_of_baselines.append(all_folds_baseline)
        list_of_300_rfe.append(all_folds_SVM)
        list_of_300_lupi.append(all_folds_LUPI)
        list_of_combined.append(all_folds_combined)

    list_of_baseline_errors =np.array([1-mean for mean in np.mean(list_of_baselines,axis=1)])
    list_of_rfe_errors = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])
    list_of_lupi_errors = np.array([1-mean for mean in np.mean(list_of_300_lupi,axis=1)])
    list_of_combined_errors = np.array([1 - mean for mean in np.mean(list_of_combined, axis=1)])

    list_of_combined_errors = list_of_combined_errors[np.argsort(list_of_baseline_errors)]
    list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_baseline_errors)]
    list_of_lupi_errors = list_of_lupi_errors[np.argsort(list_of_baseline_errors)]
    list_of_baseline_errors = list_of_baseline_errors[np.argsort(list_of_baseline_errors)]


    baseline_error_bars=list(stats.sem(list_of_baselines,axis=1))
    rfe_error_bars = list(stats.sem(list_of_300_rfe,axis=1))
    lupi_error_bars = list(stats.sem(list_of_300_lupi,axis=1))
    combined_error_bars = list(stats.sem(list_of_combined,axis=1))



    total_improvement_over_rfe, total_improvement_over_baseline, rfe_total_improvement_over_baseline = 0, 0, 0
    total_improvement_over_combined = 0
    # #########################


    lufe_vs_rfe_mean, lufe_vs_rfe_count=compare_two_results(list_of_lupi_errors, list_of_rfe_errors, 'LUFe', 'rfe')
    lufe_vs_baseline_mean, lufe_vs_baseline_count = compare_two_results(list_of_lupi_errors, list_of_baseline_errors, 'LUFe', 'baseline')
    rfe_vs_baseline_mean, rfe_vs_baseline_count = compare_two_results(list_of_rfe_errors, list_of_baseline_errors, 'RFE', 'baseline')
    lufe_vs_combined_mean, lufe_vs_combined_count = compare_two_results(list_of_lupi_errors, list_of_combined_errors, 'LUFe', 'combined')

    combined_vs_rfe_mean, combined_vs_rfe_count = compare_two_results(list_of_combined_errors,list_of_rfe_errors, 'combined', 'RFE')
    combined_vs_baseline_mean, combined_vs_baseline_count = compare_two_results(list_of_combined_errors, list_of_baseline_errors,'combined', 'baseline')
    print ('mean improvemnt vs baseline',combined_vs_baseline_mean)


    all_results_this_percentage = [lufe_vs_rfe_count, lufe_vs_baseline_count, rfe_vs_baseline_count, lufe_vs_rfe_mean, lufe_vs_baseline_mean, rfe_vs_baseline_mean, lufe_vs_combined_count, lufe_vs_combined_mean]
    print(percent_of_priv, all_results_this_percentage)
    all_settings_list.append(all_results_this_percentage)
    print(len(all_settings_list))

print(type(all_settings_list))



all_settings_list=np.array(all_settings_list)
print(all_settings_list.shape)

folder=(get_full_path('Desktop/Privileged_Data/Collected_results_NEW/'))

lupi_vs_rfe_list=all_settings_list[:,0]
print(lupi_vs_rfe_list)
np.save(folder+'lupi_vs_rfe_list-{}'.format(toporbottom),lupi_vs_rfe_list)

lupi_vs_all_list=all_settings_list[:,1]
print(lupi_vs_all_list)
np.save(folder+'lupi_vs_all_list-{}'.format(toporbottom),lupi_vs_all_list)

rfe_vs_all_list=all_settings_list[:,2]
print(rfe_vs_all_list)
np.save(folder+'rfe_vs_all_list-{}'.format(toporbottom),rfe_vs_all_list)

lupi_vs_rfe_mean=all_settings_list[:,3]
print(lupi_vs_rfe_mean)
np.save(folder+'lupi_vs_rfe_mean-{}'.format(toporbottom),lupi_vs_rfe_mean)

lupi_vs_all_mean=all_settings_list[:,4]
print(lupi_vs_all_mean)
np.save(folder+'lupi_vs_all_mean-{}'.format(toporbottom),lupi_vs_all_mean)

rfe_vs_all_mean=all_settings_list[:,5]
print(rfe_vs_all_mean)
np.save(folder+'rfe_vs_all_mean-{}'.format(toporbottom),rfe_vs_all_mean)

lupi_vs_combined_list = all_settings_list[:,6]
print(lupi_vs_combined_list)
np.save(folder+'lupi_vs_combined-{}'.format(toporbottom),lupi_vs_combined_list)

lupi_vs_combined_mean=all_settings_list[:,7]
print(lupi_vs_combined_mean)
np.save(folder+'lupi_vs_combined_mean-{}'.format(toporbottom),lupi_vs_combined_mean)

