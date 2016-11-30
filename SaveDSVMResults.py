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


method = 'RFE'
dataset='tech'
n_top_feats= 500
percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1
lufecolor='forestgreen'
rfecolor='purple'
basecolor='dodgerblue'
dsvmcolor= 'red'
###########################################################

num_datasets=295

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = 'dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

count=0
dsvm_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe1 = [], [], []
    for seed_num in range (num_repeats):
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats,seed_num))
        for inner_fold in range(num_folds):
            # if not os.path.exists(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats)):
            #     print ('datasetnum {} seednum {} inner_fold {}'.format(dataset_num,seed_num,inner_fold))
            #     count+=1
            #     print (count)
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe1+=[lupi_score]
    dsvm_lufe.append(all_folds_lufe1)


list_of_dsvm_errors = np.array([1 - mean for mean in np.mean(dsvm_lufe, axis=1)]) * 100
dsvm_error_bars_295 = list(stats.sem(dsvm_lufe, axis=1) * 100)



####################################################################### This part to get the first 40

list_of_all=[]
list_of_300_rfe=[]
list_of_300_lufe=[]
num_datasets=49

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = '10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

list_of_300_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe = [], [], []
    for seed_num in range (num_repeats):
        # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]
            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe+=[lupi_score]

    list_of_all.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lufe.append(all_folds_lufe)


list_of_baseline_errors_49 = np.array([1 - mean for mean in np.mean(list_of_all, axis=1)]) * 100
list_of_rfe_errors_49 = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])*100
list_of_lufe_errors_49 = np.array([1 - mean for mean in np.mean(list_of_300_lufe, axis=1)]) * 100

# print (list_of_baseline_errors)
baseline_error_bars_49=list(stats.sem(list_of_all, axis=1) * 100)
rfe_error_bars_49 = list(stats.sem(list_of_300_rfe,axis=1)*100)
lufe_error_bars_49 = list(stats.sem(list_of_300_lufe, axis=1) * 100)

######################################################################## This part to get the next 246

num_datasets=246

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = '246DATASETS-10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)

list_of_all=[]
list_of_300_rfe=[]
list_of_300_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe = [], [], []
    for seed_num in range (num_repeats):
        # output_directory = (get_full_path('Desktop/Privileged_Data/{}/tech{}-top{}chosen/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,seed_num)))
        output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
        for inner_fold in range(num_folds):
            with open(os.path.join(output_directory,'baseline-{}.csv'.format(inner_fold)),'r') as baseline_file:
                baseline_score = float(baseline_file.readline().split(',')[0])
                all_folds_baseline+=[baseline_score]
            with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_svm_file:
                svm_score = float(cv_svm_file.readline().split(',')[0])
                all_folds_SVM+=[svm_score]
            with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
                lupi_score = float(cv_lupi_file.readline().split(',')[0])
                all_folds_lufe+=[lupi_score]

    list_of_all.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lufe.append(all_folds_lufe)


list_of_baseline_errors_246 =np.array([1-mean for mean in np.mean(list_of_all,axis=1)])*100
list_of_rfe_errors_246 = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])*100
list_of_lufe_errors_246 = np.array([1 - mean for mean in np.mean(list_of_300_lufe, axis=1)]) * 100

# print (list_of_baseline_errors)
baseline_error_bars_246=list(stats.sem(list_of_all, axis=1) * 100)
rfe_error_bars_246 = list(stats.sem(list_of_300_rfe,axis=1)*100)
lufe_error_bars_246 = list(stats.sem(list_of_300_lufe, axis=1) * 100)
########################################################################
list_of_rfe_errors = np.hstack((list_of_rfe_errors_49,list_of_rfe_errors_246))
print(list_of_rfe_errors.shape)
list_of_lufe_errors = np.hstack((list_of_lufe_errors_49,list_of_lufe_errors_246))
list_of_all_errors = np.hstack((list_of_baseline_errors_49, list_of_baseline_errors_246))


list_of_dsvm_errors = list_of_dsvm_errors[np.argsort(list_of_all_errors)]
list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_all_errors)]
list_of_lufe_errors = list_of_lufe_errors[np.argsort(list_of_all_errors)]
list_of_all_errors = list_of_all_errors[np.argsort(list_of_all_errors)]


baseline_error_bars=np.hstack((baseline_error_bars_49,baseline_error_bars_246))
rfe_error_bars = np.hstack((rfe_error_bars_49,rfe_error_bars_246))
lufe_error_bars = np.hstack((lufe_error_bars_49,lufe_error_bars_246))


if method=='UNIVARIATE':
    method='ANOVA'


# plt.style.use('grayscale')
# plt.subplot(2,1,1)

ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

keyword='dSVM'
ax0.plot(list(range(len(list_of_rfe_errors))), list_of_lufe_errors,label='LUFe-SVM+-{}-{}'.format(method, n_top_feats),color=lufecolor)
ax0.plot(list(range(len(list_of_rfe_errors))), list_of_dsvm_errors,label='LUFe-dSVM-{}-{}'.format(method, n_top_feats),color=dsvmcolor)
ax0.legend(loc='lower right',prop={'size':10})
ax0.set_ylabel('Error rate (%)')


# ax0.errorbar(list(range(len(list_of_rfe_errors))), list_of_lufe_errors, yerr = lufe_error_bars, label='LUFe-{}-{}'.format(method, n_top_feats), markersize=3, fillstyle='none', color=lufecolor)#, marker='x',linestyle='-.')
# # ax0.errorbar(list(range(len(list_of_rfe_errors))), list_of_rfe_errors, yerr = rfe_error_bars, label='{}-{}'.format(method,n_top_feats),markersize=3,fillstyle='none',color=rfecolor)#,marker='s', linestyle='-.')
# # ax0.errorbar(list(range(len(list_of_rfe_errors))), list_of_all_errors, yerr = baseline_error_bars, label='ALL', color=basecolor)#,linestyle='--')
# ax0.errorbar(list(range(len(list_of_rfe_errors))), list_of_dsvm_errors, yerr = dsvm_error_bars_295, label='dSVM-{}-{}'.format(method, n_top_feats), markersize=3, fillstyle='none', color=dsvmcolor)#, marker='x',linestyle='-.')
# ax0.legend(loc='lower right',prop={'size':10})
# ax0.set_ylabel('Error rate (%)')

save_path = get_full_path('Desktop/all_295/DSVM-{}-{}-{}-{}-top{}'.format(keyword,lufecolor,rfecolor,basecolor,n_top_feats))
os.makedirs(save_path, exist_ok=True)


plt.savefig(os.path.join(save_path,'just_lufe_and_{}-nobars.png'.format(keyword)))


#
#
# ################################################
#
#
# def compare_two_settings(setting_one_errors,setting_two_errors,name_one,name_two):
#     improvements_list=[]
#     for error_one, error_two in zip(setting_one_errors, setting_two_errors):
#         improvements_list.append(error_one - error_two)
#     improvements_list = np.array(improvements_list)
#     print('lupi helped in', len(np.where(improvements_list > 0)[0]), 'cases vs all-feats-baseline')
#     print('mean improvement', np.mean(improvements_list))
#     with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
#         outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
#     return(improvements_list)
#
#
#
# ##########################################      PLOTTING
#
# ax1 = plt.subplot2grid((3,1), (2,0))
# # ax1.plot(y, x)
#
# def plot_figures(list_of_improvements,name1,name2,color1,color2):
#     list_of_worse = np.where(list_of_improvements < 0)[0]
#     list_of_better = np.where(list_of_improvements > 0)[0]
#     ax1 = plt.axes()
#     ax1.set_ylim(-7, 15)
#     ax1.bar(range(295),list_of_improvements)
#     ax1.bar(list_of_better, list_of_improvements[list_of_better], color=color1,)
#     ax1.bar(list_of_worse, list_of_improvements[list_of_worse], color=color2)
#     ax1.set_ylabel('Reduction in error rate \n by {} vs {}(%)'.format(name1,name2))
#     ax1.set_xlabel('Dataset index')
#     plt.savefig(get_full_path('{}/{}_vs_{}2.png'.format(save_path,name1,name2)))
#     plt.clf()
#
# ##########################################
#
# def compare_and_plot(setting_one_errors,setting_two_errors,name_one,name_two,color1,color2):
#     improvements_list = compare_two_settings(setting_one_errors,setting_two_errors,name_one,name_two)
#     plot_figures(improvements_list,name_one,name_two,color1,color2)
#
# compare_and_plot(list_of_rfe_errors,list_of_lufe_errors,'RFE','LUFe',rfecolor,lufecolor)
# compare_and_plot(list_of_all_errors, list_of_lufe_errors, 'ALL', 'LUFe', rfecolor, lufecolor)
# compare_and_plot(list_of_all_errors,list_of_rfe_errors,'ALL','RFE',rfecolor,lufecolor)
# compare_and_plot(list_of_rfe_errors,list_of_dsvm_errors,'RFE','dSVM+',rfecolor,lufecolor)
# compare_and_plot(list_of_all_errors,list_of_dsvm_errors,'ALL','dSVM+',rfecolor,lufecolor)
# compare_and_plot(list_of_lufe_errors,list_of_dsvm_errors,'LUFe','dSVM+',rfecolor,lufecolor)
#
# ##########################################
#
# total_lufe = np.sum(list_of_lufe_errors) / len(list_of_rfe_errors)
# print ('lufe', 100 - total_lufe)
#
# total_all = np.sum(list_of_all_errors) / len(list_of_rfe_errors)
# print ('all',100-total_all)
#
# total_rfe = np.sum(list_of_rfe_errors)/len(list_of_rfe_errors)
# print ('rfe',100-total_rfe)
#
# print (total_rfe - total_lufe)
# print (total_all - total_lufe)
# print (total_all-total_rfe)
#
# print ('total lufe error', total_lufe)
# print ('total rfe error', total_rfe)
# print ('total all error', total_all)
#
# print ((total_rfe-total_lufe)/total_rfe*100)
# print ((total_all-total_lufe)/total_all*100)