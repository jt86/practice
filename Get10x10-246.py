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
num_datasets=246

method = 'RFE'
dataset='tech'
n_top_feats= 300
percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1

lufecolor='forestgreen'
rfecolor='purple'
basecolor='dodgerblue'

# 246DATASETS-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = '246DATASETS-10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
# experiment_name = '10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
# print(experiment_name)
# print('10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances')
# if experiment_name == ('10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'):
#     print (True)
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)


list_of_baselines=[]
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
    if dataset_num==25:
        print ('\nbaseline\n',np.reshape(all_folds_baseline,(10,10)))
        print('\nrfe\n',np.reshape(all_folds_SVM,(10,10)))
        print('\nlufe\n', np.reshape(all_folds_lufe, (10, 10)))
    list_of_baselines.append(all_folds_baseline)
    list_of_300_rfe.append(all_folds_SVM)
    list_of_300_lufe.append(all_folds_lufe)


list_of_baseline_errors =np.array([1-mean for mean in np.mean(list_of_baselines,axis=1)])*100
list_of_rfe_errors = np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])*100
list_of_lufe_errors = np.array([1 - mean for mean in np.mean(list_of_300_lufe, axis=1)]) * 100

# print (list_of_baseline_errors)

list_of_rfe_errors = list_of_rfe_errors[np.argsort(list_of_baseline_errors)]
list_of_lufe_errors = list_of_lufe_errors[np.argsort(list_of_baseline_errors)]
list_of_baseline_errors = list_of_baseline_errors[np.argsort(list_of_baseline_errors)]


baseline_error_bars=list(stats.sem(list_of_baselines,axis=1)*100)
rfe_error_bars = list(stats.sem(list_of_300_rfe,axis=1)*100)
lufe_error_bars = list(stats.sem(list_of_300_lufe, axis=1) * 100)


if method=='UNIVARIATE':
    method='ANOVA'


# plt.style.use('grayscale')
# plt.subplot(2,1,1)

ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)


ax0.errorbar(list(range(num_datasets)), list_of_lufe_errors, yerr = lufe_error_bars, label='LUFe-{}-{}'.format(method, n_top_feats), markersize=3, fillstyle='none', color=lufecolor)#, marker='x',linestyle='-.')
ax0.errorbar(list(range(num_datasets)), list_of_rfe_errors, yerr = rfe_error_bars, label='{}-{}'.format(method,n_top_feats),markersize=3,fillstyle='none',color=rfecolor)#,marker='s', linestyle='-.')
ax0.errorbar(list(range(num_datasets)), list_of_baseline_errors, yerr = baseline_error_bars, label='ALL',color=basecolor)#,linestyle='--')
ax0.legend(loc='lower right',prop={'size':10})
ax0.set_ylabel('Error rate (%)')
plt.show()

os.makedirs(get_full_path('Desktop/246DATASETS-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances-RFE/Combined-plots-{}-{}-{}'.format(lufecolor,rfecolor,basecolor)))
save_path = get_full_path('Desktop/246DATASETS-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances-RFE/Combined-plots-{}-{}-{}'.format(lufecolor,rfecolor,basecolor))

# plt.savefig(get_full_path('Desktop/All-new-results/{}.png'.format(keyword)))
outputfile=open(get_full_path('Desktop/246results/{}.txt'.format(keyword)),'w')


lupi_improvements =0
lupi_worse = 0
total_improvement_over_rfe, total_improvement_over_baseline, total_improvement_over_baseline2 = 0,0,0
lufe_vs_rfe_improvements, lufe_vs_all_improvements, rfe_vs_all_improvements = [], [], []

for rfe_error, lupi_error in zip(list_of_rfe_errors, list_of_lufe_errors):
    total_improvement_over_rfe+=(rfe_error-lupi_error)
    if rfe_error>lupi_error:
        lupi_improvements+=1
    if rfe_error==lupi_error:
        print('same error rate!')
    else:
        lupi_worse+=1
    lufe_vs_rfe_improvements.append(rfe_error - lupi_error)

lufe_vs_rfe_improvements=np.array(lufe_vs_rfe_improvements)
print(lufe_vs_rfe_improvements)

print('lupi helped in',lupi_improvements,'cases vs standard SVM')
print('mean improvement', total_improvement_over_rfe/len(list_of_rfe_errors))
outputfile.write('\nlupi helped in {} cases vs standard SVM'.format(lupi_improvements))
outputfile.write('\nmean improvement={}'.format(total_improvement_over_rfe/len(list_of_rfe_errors)))

################################################

lupi_improvements =0
lupi_worse = 0
for baseline_error, lupi_error in zip(list_of_baseline_errors, list_of_lufe_errors):
    total_improvement_over_baseline+=(baseline_error-lupi_error)
    if baseline_error>lupi_error:
        lupi_improvements+=1
    else:
        lupi_worse+=1
    lufe_vs_all_improvements.append(baseline_error - lupi_error)
lufe_vs_all_improvements=np.array(lufe_vs_all_improvements)

print('lupi helped in',lupi_improvements,'cases vs all-feats-baseline')
print('mean improvement', total_improvement_over_baseline/len(list_of_rfe_errors))


outputfile.write('\nlupi helped in {} cases vs all-feats-baseline'.format(lupi_improvements))
outputfile.write('\nmean improvement={}'.format(total_improvement_over_baseline/len(list_of_rfe_errors)))

#############################################

rfe_improvements =0
rfe_worse = 0
for baseline_error, rfe_error in zip(list_of_baseline_errors,list_of_rfe_errors):
    total_improvement_over_baseline2+=(baseline_error-rfe_error)
    if baseline_error>rfe_error:
        rfe_improvements+=1
    else:
        rfe_worse+=1
    rfe_vs_all_improvements.append(baseline_error - rfe_error)
rfe_vs_all_improvements=np.array(rfe_vs_all_improvements)

print('feat selection helped in',rfe_improvements,'cases vs all-feats-baseline')
print('mean improvement', total_improvement_over_baseline2/len(list_of_rfe_errors))
outputfile.write('\nrfe helped in {} cases vs baseline'.format(rfe_improvements))
outputfile.write('\nmean improvement={}'.format(total_improvement_over_baseline2/len(list_of_rfe_errors)))
outputfile.close()
##########################################

# ax1 = plt.subplot2grid((3,1), (2,0))
# ax1.plot(y, x)

ax1=plt.axes()

list_of_worse = np.where(lufe_vs_rfe_improvements < 0)[0]
print('list of worse',list_of_worse)
list_of_better = np.where(lufe_vs_rfe_improvements > 0)[0]
print('list of better',list_of_better)

ax1.bar(list_of_better, lufe_vs_rfe_improvements[list_of_better], color=lufecolor)
ax1.bar(list_of_worse, lufe_vs_rfe_improvements[list_of_worse], color=rfecolor)
ax1.set_ylabel('Reduction in error rate \n by LUFe vs {}(%)'.format(method))
# ax1.set_ylim(-7,15)
ax1.set_xlabel('Dataset index')
# plt.axes('')
plt.savefig(get_full_path('{}/lufe_vs_rfe.png'.format(save_path)))
plt.show()

ax2=plt.axes()


list_of_worse2 = np.where(lufe_vs_all_improvements < 0)[0]
print('list of worse',list_of_worse2)
list_of_better2 = np.where(lufe_vs_all_improvements > 0)[0]
print('list of better',list_of_better2)

ax2.bar(list_of_better2, lufe_vs_all_improvements[list_of_better2], color=lufecolor)
ax2.bar(list_of_worse2, lufe_vs_all_improvements[list_of_worse2], color=basecolor)
ax2.set_ylabel('Reduction in error rate \n by LUFe vs ALL(%)')
ax2.set_ylim(-7,15)
ax2.set_xlabel('Dataset index')
# plt.axes('')
plt.savefig(get_full_path('{}/lufe_vs_all.png'.format(save_path)))
plt.show()

ax3=plt.axes()

list_of_worse3 = np.where(rfe_vs_all_improvements < 0)[0]
print('list of worse',list_of_worse3)
list_of_better3 = np.where(rfe_vs_all_improvements > 0)[0]
print('list of better',list_of_better3)

ax3.bar(list_of_better3, rfe_vs_all_improvements[list_of_better3], color=rfecolor)
ax3.bar(list_of_worse3, rfe_vs_all_improvements[list_of_worse3], color=basecolor)
ax3.set_ylabel('Reduction in error rate \n by {} vs ALL(%)'.format(method))
ax3.set_ylim(-7,15)
ax3.set_xlabel('Dataset index')
# plt.axes('')
plt.savefig(get_full_path('{}/rfe_vs_all.png'.format(save_path)))
plt.show()




total_lufe = np.sum(list_of_lufe_errors) / num_datasets
print ('lufe', 100 - total_lufe)

total_all = np.sum(list_of_baseline_errors)/num_datasets
print ('all',100-total_all)

total_rfe = np.sum(list_of_rfe_errors)/num_datasets
print ('rfe',100-total_rfe)

print (total_rfe - total_lufe)
print (total_all - total_lufe)
print (total_all-total_rfe)

print ('total lufe error', total_lufe)
print ('total rfe error', total_rfe)
print ('total all error', total_all)

print ((total_rfe-total_lufe)/total_rfe*100)