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
import pandas

num_repeats = 10
num_folds = 10
method = 'RFE'
dataset='tech'
percentofinstances=100
toporbottom='top'
step=0.1

class Setting:
    def __init__(self,num_datasets,classifier_type,n_top_feats,c_value,percent_of_priv,featsel):
        self.num_datasets = num_datasets
        self.classifier_type = classifier_type
        self.n_top_feats = n_top_feats
        self.c_value = c_value
        self.percent_of_priv = percent_of_priv

        self.featsel = featsel

        self.name = '{}-{}-{}{}'.format(classifier_type, c_value, percent_of_priv,featsel)

def get_errors(setting):
    scores = get_scores(setting)
    return (np.array([1-mean for mean in np.mean(scores,axis=1)])*100)


def get_scores(setting):
    if setting.featsel != '':
        featsel = '-{}'.format(setting.featsel)
    else:
        featsel=''
    return np.load(get_full_path('Desktop/SavedNPArrayResults/{}/{}-{}-{}-{}-{}{}.npy'.format(dataset,setting.num_datasets,setting.classifier_type,setting.n_top_feats,setting.c_value,setting.percent_of_priv,featsel)))


def plot_bars(setting_one,setting_two):
    name_one, name_two = setting_one.name, setting_two.name
    improvements_list = get_improvements_list(setting_one,setting_two)
    plt.bar(range(len(improvements_list)),improvements_list, color='black')
    plt.title('{} vs {} \n Improvement by {} = {}%, {} of {} cases'.format(name_one, name_two, name_two, round(np.mean(improvements_list),2),len(np.where(improvements_list > 0)[0]),len(improvements_list)))
    plt.ylabel('<---{} better  (%)   {} better--->'.format(name_one,name_two))
    plt.xlabel('dataset index')
    plt.ylim(-10,15)
    plt.savefig(get_full_path('Desktop/SavedNPArrayResults/tech/plots/{}_VS_{}'.format(name_one,name_two)))
    plt.show()

num_datasets=295


all_baseline = Setting(295,'baseline',300,'cross-val',100, '')

def plot_total_comparison(setting_one,setting_two, baseline_setting=all_baseline):
    setting_one_errors = get_errors(setting_one)
    setting_two_errors = get_errors(setting_two)
    baseline_errors =  get_errors(baseline_setting)
    setting_one_errors=setting_one_errors[np.argsort(baseline_errors[:num_datasets])]
    setting_two_errors = setting_two_errors[np.argsort(baseline_errors[:num_datasets])]
    baseline_errors = baseline_errors[np.argsort(baseline_errors[:num_datasets])]
    plt.plot(range(num_datasets),setting_one_errors[:num_datasets],color='blue',label=setting_one.name)
    plt.plot(range(num_datasets), setting_two_errors[:num_datasets],color='red',label=setting_two.name)
    plt.plot(range(num_datasets), baseline_errors[:num_datasets], color='black', label='all feats baseline')
    plt.ylabel('Error (%)')
    plt.xlabel('Dataset number (sorted)')
    plt.legend(loc='best')
    plt.savefig(get_full_path('Desktop/SavedNPArrayResults/tech/plots/TOTALERROR{}VS{}'.format(setting_one.name, setting_two.name)))
    plt.show()

def get_improvements_list(setting_one, setting_two):
    setting_one_errors = get_errors(setting_one)
    setting_two_errors = get_errors(setting_two)
    improvements_list = []
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two)
    return np.array(improvements_list)

def compare_two_settings(setting_one, setting_two):
    improvements_list = get_improvements_list()
    name_one, name_two = setting_one.name, setting_two.name
    plot_bars(improvements_list,name_one,name_two)
    plot_total_comparison(setting_one, setting_two)
    print('{} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),len(setting_one_errors),np.mean(improvements_list)))
    return(improvements_list)


def get_errors_single_fold(setting):
    if setting.featsel != '':
        featsel = '-{}'.format(setting.featsel)
    scores = np.load(get_full_path('Desktop/SavedNPArrayResults/{}/{}-{}-{}-{}-{}.npy'.format(dataset,setting.num_datasets,setting.classifier_type,setting.n_top_feats,setting.c_value,featsel)))
    errors = np.array([1-score for score in scores])*100
    return errors

def compare_two_settings_ind_folds(setting_one, setting_two):
    setting_one_errors = get_errors_single_fold(setting_one)
    setting_two_errors = get_errors_single_fold(setting_two)
    name_one,name_two = setting_one.name, setting_two.name
    improvements_list=[]
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two)
    improvements_list = np.array(improvements_list)
    shape = setting_one_errors.shape[0]*setting_one_errors.shape[1]
    print('{} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),shape,np.mean(improvements_list)))
    # with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
    #     outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
    return(improvements_list)




#
# lufe_baseline = Setting(295,'lupi',300,'cross-val',100, '')
#
# svm_baseline = Setting(295,'svm',300,'cross-val',100, '')
# dsvm_crossval = Setting(295,'dsvm',300,'cross-val',100, '')
# dsvm_top10 =  Setting(295,'dsvm',300,'cross-val',10, '')
# dsvm_top50 = Setting(295,'dsvm',300,'cross-val',50, '')
#
# top_10_lufe_295 = Setting(295, 'lupi', 300, 'cross-val', 10, '')
# top_25_lufe_295 = Setting(295, 'lupi', 300, 'cross-val', 25, '')
# top_50_lufe_295 = Setting(295, 'lupi', 300, 'cross-val', 50, '')

# top_228_anova_228 = Setting(228, 'lupiANOVA', 300, 'cross-val', 100, '')

# compare_two_settings(top_50_lufe_295, lufe_baseline)

# anova_lupi = Setting(295, 'lupi', 300, 'cross-val', 100, 'anova')
# anova_svm = Setting(295, 'svm', 300, 'cross-val', 100, 'anova')
# chi2_lupi = Setting(295, 'lupi', 300, 'cross-val', 100, 'chi2')
# chi2_svm = Setting(295, 'svm', 300, 'cross-val', 100, 'chi2')


# mutinfo_lupi_10 = Setting(240, 'lupi', 300, 'cross-val', 10, 'mutinfo')

mutinfo_lupi_100 = Setting(295, 'lupi', 300, 'cross-val', 100, 'mutinfo')
mutinfo_svm = Setting(295,'svm',300,'cross-val',100,'mutinfo')




##########  COMPARING FIXED C WITH CROSS-VALIDATED ############

# print('Comparing dSVM+ LUFe and SVM+ LUFe...')
# for c in [1,10,100,1000]:
#     print('\n C = {}'.format(c))
#     for percent_of_priv in [10,25,50,75,100]:
#         dsvm = Setting(295,'lupi',300,c,percent_of_priv)
#         compare_two_settings(lufe_baseline,dsvm)
# print('\n Comparing dSVM+ LUFe and standard RFE...')
#
# for c in [1,10,100,1000]:
#     print('\n C = {}'.format(c))
#     for percent_of_priv in [10,25,50,75,100]:
#         dsvm = Setting(295,'lupi',300,c,percent_of_priv)
#         compare_two_settings(svm_baseline,dsvm)
# print('\n Comparing dSVM+ LUFe and all-features SVM...')
# for c in [1, 10, 100, 1000]:
#     print('\n C = {}'.format(c))
#     for percent_of_priv in [10, 25, 50, 75, 100]:
#         dsvm = Setting(295, 'lupi', 300, c, percent_of_priv)
#         compare_two_settings(all_baseline, dsvm)
# dsvm_lufe_1,dsvm_lufe_10,dsvm_lufe_100,dsvm_lufe_1000=[],[],[],[]
# for dataset_num in range(190):
#     all_folds_lufe1, all_folds_lufe10, all_folds_lufe100, all_folds_lufe1000 = [],[],[],[]
#     for seed_num in range (num_repeats):
#         output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-FIXEDC-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats, seed_num))
#         for inner_fold in range(num_folds):
#             with open(os.path.join(output_directory,'lupi-{}-{}-C=1.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
#                 all_folds_lufe1+=[float(cv_lupi_file.readline().split(',')[0])]
#             with open(os.path.join(output_directory,'lupi-{}-{}-C=10.csv').format(inner_fold, n_top_feats),'r') as cv_lupi_file:
#                 all_folds_lufe10 += [float(cv_lupi_file.readline().split(',')[0])]
#             with open(os.path.join(output_directory,'lupi-{}-{}-C=100.csv').format(inner_fold,n_top_feats),'r') as cv_lupi_file:
#                 all_folds_lufe100+=[float(cv_lupi_file.readline().split(',')[0])]
#             with open(os.path.join(output_directory,'lupi-{}-{}-C=1000.csv').format(inner_fold, n_top_feats),'r') as cv_lupi_file:
#                 all_folds_lufe1000 += [float(cv_lupi_file.readline().split(',')[0])]
##     dsvm_lufe_1.append(all_folds_lufe1)
#     dsvm_lufe_10.append(all_folds_lufe10)
#     dsvm_lufe_100.append(all_folds_lufe100)
#     dsvm_lufe_1000.append(all_folds_lufe1000)


