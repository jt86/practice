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
    print('\n\n')
    list1_better, list2_better = 0,0
    for error1, error2 in zip (list1,list2):
        if error1 > error2:
            list2_better += 1
        else:
            list1_better += 1
    print ('{} helped in {} cases vs {}'.format(name1,list1_better,name2))
    print ('{} was worse in {} cases vs {}'.format(name1,list2_better,name2))

    mean = (np.mean(list1)-np.mean(list2))
    return (mean,list1_better)


all_settings_list=[]
for percent_of_priv in list(range(10,90,10)):
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

    print (list_of_combined_errors)
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


    mean1, lupi_improvements=compare_two_results(list_of_rfe_errors,list_of_lupi_errors,'rfe','LUFe')
    #
    # #########################
    #
    # lupi_improvements2 =0
    # lupi_worse2 = 0
    # for baseline_error, lupi_error in zip(list_of_baseline_errors,list_of_lupi_errors):
    #     total_improvement_over_baseline+=(baseline_error-lupi_error)
    #     if baseline_error>lupi_error:
    #         lupi_improvements2+=1
    #     else:
    #         lupi_worse2+=1
    # print('lupi helped in',lupi_improvements2,'cases vs all-feats-baseline')
    # print('mean improvement', total_improvement_over_baseline/len(list_of_rfe_errors))
    # mean2=total_improvement_over_baseline/len(list_of_rfe_errors)
    #
    # #########################
    #
    # rfe_improvements =0
    # rfe_worse = 0
    # for baseline_error, rfe_error in zip(list_of_baseline_errors,list_of_rfe_errors):
    #     rfe_total_improvement_over_baseline+=(baseline_error - rfe_error)
    #     if baseline_error>rfe_error:
    #         rfe_improvements+=1
    #     else:
    #         rfe_worse+=1
    # print('feat selection helped in',rfe_improvements,'cases vs all-feats-baseline')
    # print('mean improvement', rfe_total_improvement_over_baseline / len(list_of_rfe_errors))
    # mean3 = rfe_total_improvement_over_baseline / len(list_of_rfe_errors)
    #
    # #########################

    # improvements_over_combined = 0
    # worse_than_combined = 0
    #
    # for combined_error, lupi_error in zip(list_of_combined_errors,list_of_lupi_errors):
    #     total_improvement_over_combined+=(combined_error-lupi_error)
    #     if combined_error>lupi_error:
    #         improvements_over_combined+=1
    #     else:
    #         worse_than_combined+=1
    # #
    # print('lupi helped in',improvements_over_combined,'cases vs combined')
    # print('lupi was worse than combined in',worse_than_combined,'cases')
    # mean4 = total_improvement_over_combined/len(list_of_rfe_errors)
    # print('mean improvement',mean4)
    # total_improvement_over_combined2=np.mean(list_of_combined_errors)-np.mean(list_of_lupi_errors)
    # print(total_improvement_over_combined2)
    #
    # print('\n')
    # compare_two_results(list_of_combined_errors,list_of_lupi_errors,'combined','LUFe')

    ##########################


    #
    # single_setting_list = [lupi_improvements,lupi_improvements2,rfe_improvements,mean1,mean2,mean3,improvements_over_combined,mean4]
    # print(percent_of_priv,single_setting_list)
    # all_settings_list.append(single_setting_list)
    # print(all_settings_list)

# print(type(all_settings_list))
#
#
#
# all_settings_list=np.array(all_settings_list)
# print(all_settings_list.shape)
#
# folder=(get_full_path('Desktop/Privileged_Data/Collected_results_NEW/'))
#
# lupi_vs_rfe_list=all_settings_list[:,0]
# print(lupi_vs_rfe_list)
# np.save(folder+'lupi_vs_rfe_list-{}'.format(toporbottom),lupi_vs_rfe_list)
#
# lupi_vs_all_list=all_settings_list[:,1]
# print(lupi_vs_all_list)
# np.save(folder+'lupi_vs_all_list-{}'.format(toporbottom),lupi_vs_all_list)
#
# rfe_vs_all_list=all_settings_list[:,2]
# print(rfe_vs_all_list)
# np.save(folder+'rfe_vs_all_list-{}'.format(toporbottom),rfe_vs_all_list)
#
# lupi_vs_rfe_mean=all_settings_list[:,3]
# print(lupi_vs_rfe_mean)
# np.save(folder+'lupi_vs_rfe_mean-{}'.format(toporbottom),lupi_vs_rfe_mean)
#
# lupi_vs_all_mean=all_settings_list[:,4]
# print(lupi_vs_all_mean)
# np.save(folder+'lupi_vs_all_mean-{}'.format(toporbottom),lupi_vs_all_mean)
#
# rfe_vs_all_mean=all_settings_list[:,5]
# print(rfe_vs_all_mean)
# np.save(folder+'rfe_vs_all_mean-{}'.format(toporbottom),rfe_vs_all_mean)
#
# lupi_vs_combined_list = all_settings_list[:,6]
# print(lupi_vs_combined_list)
# np.save(folder+'lupi_vs_combined-{}'.format(toporbottom),lupi_vs_combined_list)
#
# lupi_vs_combined_mean=all_settings_list[:,7]
# print(lupi_vs_combined_mean)
# np.save(folder+'lupi_vs_combined_mean-{}'.format(toporbottom),lupi_vs_combined_mean)

