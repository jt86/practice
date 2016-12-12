import os
from Get_Full_Path import get_full_path
from matplotlib import pyplot as plt
import numpy as np
num_datasets=295
from collections import defaultdict

######################## This part to get min,max,mean,stdev of deviation values in a given dataset over 10 folds

mins,maxes,means,stdevs=[],[],[],[]
output_directory = get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE')
for datasetnum in range(num_datasets):
    with open(os.path.join(output_directory, 'tech{}-dScore.csv'.format(datasetnum)) ,'r') as savefile:
        lines = savefile.readlines()
        means.append(float(lines[3].split(',')[1].strip('\n')))
        maxes.append(float(lines[4].split(',')[1].strip('\n')))
        mins.append(float(lines[5].split(',')[1].strip('\n')))
        stdevs.append(float(lines[6].split(',')[1].strip('\n')))

# print (means)
# print (maxes)
# print (mins)
# print (stdevs)


######################## This part to PLOT  min,max,mean,stdev of deviation values in a given dataset over 10 folds

# plt.errorbar(range(num_datasets),means,yerr=stdevs, label='mean with stdev')
# plt.scatter(range(num_datasets), maxes, label='maxes',color='green',marker='.')
# plt.scatter(range(num_datasets), mins, label='min',color='red',marker='.')
# # plt.legend(loc='best')
# plt.title('Size of deviation values for training data (1 − yi[(w,xi)+b]')
# plt.xlabel('Dataset number')
# plt.ylim(-7,7)
# plt.show()


######################## This part to count how many times each c value occurs

# counts=defaultdict(int)
# print(counts)
# output_directory = get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE')
# for datasetnum in range(num_datasets):
#     with open(os.path.join(output_directory, 'cross-validation1/C_fullset-crossvalid-{}-300.txt'.format(datasetnum)) ,'r') as savefile:
#         lines=(savefile.readlines())
#         c_value=(lines[1].split(']')[1].strip('\n'))
#         counts[c_value]+=1
# print(counts)
#
# list_of_pairs =[]
# for c_value,count in counts.items():
#     print (c_value,count)
#     list_of_pairs.append([c_value,count])
# list_of_pairs.sort()
# print(list_of_pairs)


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
dsvmcolor= 'red'
from scipy import stats




####################################### THIS PART TO GET Cs

c_values=[]
# print(c_values)
output_directory = get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE')
for datasetnum in range(num_datasets):
    with open(os.path.join(output_directory, 'cross-validation1/C_fullset-crossvalid-{}-300.txt'.format(datasetnum)) ,'r') as savefile:
        lines=(savefile.readlines())
        c_value=float(lines[1].split(']')[1].strip('\n'))
        c_values.append(c_value)
# print(c_values)

c_values=np.array(c_values)

stdevs=[]
means=[]
output_directory = get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE')
for datasetnum in range(num_datasets):
    with open(os.path.join(output_directory, 'tech{}-dScore.csv'.format(datasetnum)) ,'r') as savefile:
        lines = savefile.readlines()
        means.append(float(lines[3].split(',')[1].strip('\n')))
        stdevs.append(float(lines[6].split(',')[1].strip('\n')))
        # if float(lines[6].split(',')[1].strip('\n'))>1:
        #     print(c_value,datasetnum)

# print(len(c_values),len(means))
#
# fig=plt.figure()
# ax=plt.gca()
#
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.scatter(c_values,stdevs)
# ax.set_ylim(-100,100)
# plt.show()

####################################### THIS PART TO GET dSVM ACCURACIES

num_datasets=295

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = 'dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)
seed_num, inner_fold = 1, 1
count=0
dsvm_lufe=[]
for datasetnum in range(num_datasets):
    # print(datasetnum)
    all_folds_baseline, all_folds_SVM, all_folds_lufe1 = [], [], []
    output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(datasetnum, n_top_feats,seed_num))
    with open(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats),'r') as rfe_file:
        rfe_score = float(rfe_file.readline().split(',')[0])
        # print(rfe_score)
        all_folds_lufe1+=[rfe_score]
    dsvm_lufe.append(all_folds_lufe1)


list_of_dsvm_errors = np.array([1 - mean for mean in np.mean(dsvm_lufe, axis=1)]) * 100
# dsvm_error_bars_295 = list(stats.sem(dsvm_lufe, axis=1) * 100)

# print (list_of_dsvm_errors)
# print (len(list_of_dsvm_errors))

# plt.scatter(means,list_of_dsvm_errors)
# plt.show()

####################################### THIS PART TO GET SVM+ ACCURACIES

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

seed_num=1
inner_fold=1

list_of_300_lufe=[]
for dataset_num in range(num_datasets):
    all_folds_lufe = []
    output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
    with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as rfe_file:
        rfe_score = float(rfe_file.readline().split(',')[0])
        print('lupi score', rfe_score)
        list_of_300_lufe.append(rfe_score)

########################################### This part to get the next 246
num_datasets=246
experiment_name = '246DATASETS-10x10-{}-ALLCV-3to3-featsscaled-step{}-{}{}percentpriv-{}percentinstances-{}'.format(dataset,step,percent_of_priv,toporbottom,percentofinstances,method)
keyword = '{}-{}feats-{}-3to3-{}{}instances-{}priv-step01'.format(dataset,n_top_feats,method,toporbottom,percentofinstances,percent_of_priv)
np.set_printoptions(linewidth=132)
for dataset_num in range(num_datasets):
    all_folds_baseline, all_folds_SVM, all_folds_lufe = [], [], []
    output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/{}/tech{}/top{}chosen-{}percentinstances/cross-validation{}/'.format(experiment_name,dataset_num,n_top_feats,percentofinstances,seed_num))
    with open(os.path.join(output_directory,'svm-{}-{}.csv').format(inner_fold,n_top_feats),'r') as rfe_file:
        rfe_score = float(rfe_file.readline().split(',')[0])
        list_of_300_lufe.append(rfe_score)

list_of_300_lufe = np.array([1 - item for item in list_of_300_lufe]) * 100
print (list_of_300_lufe.shape)

#########################################################
differences=[]
#  Positive value in differences means dsvm did better!

for dsvm_score, rfe_score in zip(list_of_dsvm_errors, list_of_300_lufe):
    difference = dsvm_score - rfe_score
    differences.append(difference)
    print (difference)
print(differences)
# print(count(np.where(differences>0)))
differences=np.array(differences)
print(len(np.where(differences > (0))[0]))
print(len(np.where(differences < (0))[0]))
print(len(np.where(differences == (0))[0]))

# fig1,ax1=plt.subplots()
# ax1.set_xscale('log')
# ax1.scatter(list_of_dsvm_errors,list_of_300_lufe)
# plt.scatter(means,differences)
# plt.yscale='log'
# plt.show()


# for item in differences
#     print(type(item))

sorted_indices = np.argsort(differences)
print(sorted_indices)


# Differences

print (differences[38])
print (differences[136])

print(differences[241])
print(differences[215])

print(c_values[sorted_indices][-20:])
print(c_values[sorted_indices][:20])



### Plot ranking in terms of improvement vs RFE, vs C parameter chosen. No pattern
fig1,ax1=plt.subplots()
ax1.set_yscale('log')
plt.scatter(range(295),c_values[sorted_indices])
plt.show()

### Plot improvement vs RFE, vs C parameter chosen. No pattern
fig2,ax2=plt.subplots()
ax2.set_yscale('log')
plt.scatter(differences,c_values)
plt.show()
