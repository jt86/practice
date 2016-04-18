'''
Module used in comparing coefficients
'''

__author__ = 'jt306'
import os
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
import numpy as np
import sys
dataset='tech'
cmin=-3
cmax=3
stepsize=0.1
topk=300



num_datasets_to_use = 49

#
#
# all_weights = []
# for datasetnum in range(49):
#     output_directory = get_full_path(('Desktop/Privileged_Data/GetScore-{}{}-{}to{}-{}-{}-tech{}').format(dataset,datasetnum,cmin,cmax,stepsize,topk,datasetnum))
#     for k in range(10):
#         for skfseed in range(10):
#             with open(os.path.join(output_directory,'normal-all-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as normal_chi2_file:
#                 all_weights+=normal_chi2_file.readline()
#
# print (all_weights.shape)


#
all_normal_coefficients, priv_coefficients = [],[]

#put all normal (ie top 300) coefficients) into all_normal_coefficients
for datasetnum in range(num_datasets_to_use):
    normal_coefs_for_one_dataset = []
    coefficients_directory = get_full_path(('Desktop/Privileged_Data/Coefficients-PenultimateIteration/Coefficients{}{}-{}to{}-{}-{}').format(dataset,datasetnum,cmin,cmax,stepsize,topk))
    for k in range(10):
        for skfseed in range(10):
            with open(os.path.join(coefficients_directory,'normal-coefficients-{}-{}.csv'.format(k,skfseed)),'r') as normal_chi2_file:
                list_of_coefs = (normal_chi2_file.readline().split(',')[:-1])
                normal_coefs_for_one_dataset+=[list_of_coefs]
    all_normal_coefficients+=[normal_coefs_for_one_dataset]
    print(len(all_normal_coefficients))
all_normal_coefficients=(np.array(all_normal_coefficients,dtype=float))
print ('all shape',all_normal_coefficients.shape)

#put remaining features (not top 300) into all_priv_coefficients
for datasetnum in range(num_datasets_to_use):
    priv_coefs_for_one_dataset = []
    coefficients_directory = get_full_path(('Desktop/Privileged_Data/Coefficients-PenultimateIteration/Coefficients{}{}-{}to{}-{}-{}').format(dataset,datasetnum,cmin,cmax,stepsize,topk))
    for k in range(10):
        for skfseed in range(10):
            # print('dataset',datasetnum,'k',k,'seed',skfseed)
            with open(os.path.join(coefficients_directory,'priv-coefficients-{}-{}.csv'.format(k,skfseed)),'r') as normal_chi2_file:
                list_of_coefs = (normal_chi2_file.readline().split(',')[:-1])
                priv_coefs_for_one_dataset+=[list_of_coefs]
    priv_coefficients+=[priv_coefs_for_one_dataset]
    print(len(priv_coefficients))
all_priv_coefficients=[]
priv_coefficients=(np.array(priv_coefficients))
for single_dataset in priv_coefficients:
    all_folds_one_dataset=[]
    for single_fold in single_dataset:
        single_fold = np.array(single_fold,dtype=float)
        all_folds_one_dataset.append(single_fold)
    all_folds_one_dataset=np.array(all_folds_one_dataset)
    print (all_folds_one_dataset.shape)
    all_priv_coefficients.append(all_folds_one_dataset)
print ('all datasets',len(all_priv_coefficients))






lupi_better = (np.load(get_full_path('Desktop/Privileged_Data/all-results/lupi_better-10x10-ALLCV-3to3-featsscaled-300.npy')))
rfe_better = (np.load(get_full_path('Desktop/Privileged_Data/all-results/rfe_better-10x10-ALLCV-3to3-featsscaled-300.npy')))
print (lupi_better)
print (lupi_better.shape)
print (rfe_better)
print (rfe_better.shape)





lupi_better_scores = []
rfe_better_scores = []
list_of_ratios = []

for datasetnum, (normal,priv) in enumerate(zip(all_normal_coefficients,all_priv_coefficients)):
    print('\n datasetnum',datasetnum)
    print('normal',normal.shape,'priv',priv.shape)
    normal_means = np.mean(np.abs(normal),axis=0)
    priv_means = np.mean(np.abs(priv),axis=0)
    normal_mean = np.mean(normal_means)
    priv_mean = np.mean(priv_means)
    normal_std = np.std(normal_means)
    priv_std = np.std(priv_means)

    print('normal:', normal_mean, '+/-', normal_std)
    print('priv:', priv_mean, '+/-', priv_std)

    if datasetnum in lupi_better:
        print('dataset',datasetnum, 'LUPI helps')
        lupi_better_scores.append([normal_mean,normal_std,priv_mean,priv_std,int(datasetnum)])

    if datasetnum in rfe_better:
        print('dataset',datasetnum, 'LUPI doesnt help - rfe better')
        rfe_better_scores.append([normal_mean,normal_std,priv_mean,priv_std,int(datasetnum)])

    # ratio =



##### This part to make scatter plot of mean coefficients (top 300 vs rest)

# lupi_better_scores = np.array(lupi_better_scores)
# rfe_better_scores = np.array(rfe_better_scores)
#
# fig = plt.figure()
# ax = fig.add_subplot(2,1,1)
#
# ax.scatter(rfe_better_scores[:,0],rfe_better_scores[:,2],color='blue',label='LUPI helps')
# ax.scatter(lupi_better_scores[:,0],lupi_better_scores[:,2],color='red',label='LUPI doesnt help')
# plt.xlabel('Mean coefficient for normal features (top 300)')
# plt.ylabel('Mean coefficient for privileged features')
# plt.xlim([0.002,0.014])
# plt.ylim([0.0,0.0009])
# # ax.set_yscale('log')
# # ax.set_xscale('log')
#
# for index, item in enumerate(lupi_better_scores):
#     plt.annotate(s=int(item[4]), xy=(item[0],item[2]))
# for index, item in enumerate(rfe_better_scores):
#     plt.annotate(s=int(item[4]), xy=(item[0],item[2]))
#
#
# plt.show()
#
#

####### This part makes OLD 49 bar graphs, showing rfe ranking vs coefficient value (red for privileged, blue for normal)

for datasetnum, (normal,priv) in enumerate(zip(all_normal_coefficients,all_priv_coefficients)):
    normal = -np.sort(-np.abs(np.mean(normal,axis=0)))
    priv = -np.sort(-np.abs(np.mean(priv,axis=0)))
    plt.bar(list(range(300)),normal,color='red',label='Top 300 features', hold=False,edgecolor='none')
    plt.bar(list(range(300,800)),priv[:500],color='blue',label='privileged features',edgecolor='none')
    plt.ylim(0,0.15)
    if datasetnum in lupi_better:
        title = 'Useful Privileged Information'
    else:
        title = 'Unhelpful Privileged Information'
    plt.title('TechTC{} - {}'.format(datasetnum,title))
    plt.savefig('fixed-axis-coefficientsplot-tech{}'.format(datasetnum))
    # plt.show()

