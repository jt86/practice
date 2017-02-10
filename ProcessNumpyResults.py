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
    def __init__(self,num_datasets,classifier_type,n_top_feats,c_value,percent_of_priv):
        self.num_datasets = num_datasets
        self.classifier_type = classifier_type
        self.n_top_feats = n_top_feats
        self.c_value = c_value
        self.percent_of_priv = percent_of_priv
        self.name = '{}-{}-{}'.format(classifier_type,c_value,percent_of_priv)


def get_errors(setting):
    scores = np.load(get_full_path('Desktop/SavedNPArrayResults/{}/{}-{}-{}-{}-{}.npy'.format(dataset,setting.num_datasets,setting.classifier_type,setting.n_top_feats,setting.c_value,setting.percent_of_priv)))
    errors = (np.array([1-mean for mean in np.mean(scores,axis=1)])*100)
    return errors



def plot_bars(improvements_list):
    # print('shape',improvements_list.shape)
    # print(np.where(improvements_list>0))
    # # print('shape',np.where(improvements_list > 0))
    # print((improvements_list[improvements_list>0])[0].shape)
    # plt.bar(np.where(improvements_list>0),improvements_list[improvements_list>0][0])
    plt.bar(range(295),improvements_list)
    plt.show()

def compare_two_settings(setting_one, setting_two):
    setting_one_errors = get_errors(setting_one)
    setting_two_errors = get_errors(setting_two)
    name_one,name_two = setting_one.name, setting_two.name
    improvements_list=[]
    for error_one, error_two in zip(setting_one_errors, setting_two_errors):
        improvements_list.append(error_one - error_two)
    improvements_list = np.array(improvements_list)
    plot_bars(improvements_list)
    print('{} vs {}: {} helped in {} of {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),len(setting_one_errors),np.mean(improvements_list)))
    # with open(os.path.join(save_path, '{}.txt'.format(keyword)), 'a') as outputfile:
    #     outputfile.write('\n{} vs {}: {} helped in {} cases, mean improvement={}%'.format(name_two,name_one,name_two,len(np.where(improvements_list > 0)[0]),np.mean(improvements_list)))
    return(improvements_list)





def get_errors_single_fold(setting):
    scores = np.load(get_full_path('Desktop/SavedNPArrayResults/{}/{}-{}-{}-{}-{}.npy'.format(dataset,setting.num_datasets,setting.classifier_type,setting.n_top_feats,setting.c_value,setting.percent_of_priv)))
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





lufe_baseline = Setting(295,'lupi',300,'cross-val',100)
all_baseline = Setting(295,'baseline',300,'cross-val',100)
svm_baseline = Setting(295,'svm',300,'cross-val',100)
dsvm_crossval = Setting(295,'dsvm',300,'cross-val',100)
dsvm_top10 =  Setting(295,'dsvm',300,'cross-val',10)
dsvm_top50 = Setting(295,'dsvm',300,'cross-val',50)
# compare_two_settings(all_baseline,lufe_baseline)
# compare_two_settings(svm_baseline,lufe_baseline)#, 'svm baseline','lufe baseline')


# compare_two_settings_ind_folds(lufe_baseline,dsvm_crossval)
compare_two_settings(dsvm_crossval,dsvm_top10)
compare_two_settings(dsvm_crossval,dsvm_top50)
compare_two_settings(dsvm_top10,all_baseline)
# print(compare_two_settings(lufe_baseline,dsvm_top10))

# print(get_errors(dsvm_top10))


# dsvm_errors = get_errors(dsvm_crossval)
# print ((dsvm_errors.shape),dsvm_errors)
#
# compare_two_settings(dsvm_crossval,lufe_baseline)
# compare_two_settings(dsvm_crossval,all_baseline)
# compare_two_settings(svm_baseline,dsvm_crossval)




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
#
# for c in [1, 10, 100, 1000]:
#     print('\n C = {}'.format(c))
#     for percent_of_priv in [10, 25, 50, 75, 100]:
#         dsvm = Setting(295, 'lupi', 300, c, percent_of_priv)
#         compare_two_settings(all_baseline, dsvm)






        #
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
#
#
#
#     dsvm_lufe_1.append(all_folds_lufe1)
#     dsvm_lufe_10.append(all_folds_lufe10)
#     dsvm_lufe_100.append(all_folds_lufe100)
#     dsvm_lufe_1000.append(all_folds_lufe1000)
#
#

