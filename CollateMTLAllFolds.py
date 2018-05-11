import os
from Get_Full_Path import get_full_path
import csv
import pandas
import numpy as np
from CollateResults import collate_single_dataset,collate_all
from ExperimentSetting import Experiment_Setting
from matplotlib import pyplot as plt
import sys

#### get svm baseline

s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='nolufe',
                       featsel='rfe', classifier='featselector', stepsize=0.1)
svm_results = np.mean(collate_all(s1), axis=1)
# svm_results = svm_results[:,[-1]].flatten()

indices = (np.argsort(svm_results))
svm_results = svm_results[indices]
#
#
#### get rfe-lufe baseline

s2 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='svmplus',
                       featsel='rfe', classifier='lufe', stepsize=0.1)

rfe_lufe_results = np.mean(collate_all(s2), axis=1)[indices]





# #### get mi-lufe baseline
#
# s3 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
#                        cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                        take_top_t='top', lupimethod='svmplus',
#                        featsel='mi', classifier='lufe', stepsize=0.1)
#
# mi_lufe_results = collate_all_datasets(s3)
# mi_lufe_results = mi_lufe_results[:, [-1]].flatten()
#
#
#
# #### get mi-lufe baseline
#
# s4 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
#                        cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                        take_top_t='top', lupimethod='svmplus',
#                        featsel='anova', classifier='lufe', stepsize=0.1)
#
# anova_lufe_results = collate_all_datasets(s4)
# anova_lufe_results = anova_lufe_results[:, [-1]].flatten()

#
#### load NN results and re-order using
names_list = []
results_dict = {}
# results_dict['svm_setting']=np.array(svm_results)[indices]
# results_dict['rfe_lufe_results']=np.array(rfe_lufe_results)[indices]


#######

def collate_nn_single_setting(filename, featsel, indices=indices):
    nn_results = []
    for foldnum in range(10):
        dataset_results = []
        with open(get_full_path('Desktop/Privileged_Data/MTL_{}_results/{}-fold{}.csv'.format(featsel,filename,foldnum)), 'r') as results_file:
            reader = csv.reader(results_file,delimiter=',')
            for count, line in enumerate(reader):
                dataset_results.append(float(line[-1]))
        nn_results.append(dataset_results)

    nn_results = np.array(nn_results)
    print('nn',nn_results.shape)
    mean_results = np.mean(nn_results,axis=0)
    print(mean_results)
    print('mean', mean_results.shape)
    print('indices',indices.shape)
    sorted_results = mean_results[indices]
    print(sorted_results.shape)
    return sorted_results



def collate_for_multiple_settings(weight,featsel):
    list_of_featnums,list_of_results = [],[]
    # for num_unsel_feats in [1,2,3]+[item for item in range(10,300,10)]+[item for item in range (1000,2200,100)]:#+['all']:
    for num_unsel_feats in [300]:
        sorted_results =collate_nn_single_setting('MTLresultsfile-3200units-weight{}-numfeats={}-learnrate0.0001'.format(weight, num_unsel_feats), featsel)
        # collate_nn_results('MTLresultsfile-3200units-weight0.001-numfeats={}-learnrate0.0001'.format(num_unsel_feats))
        list_of_featnums.append(num_unsel_feats)
        list_of_results.append(sorted_results)
    # list_of_featnums[-1] = 21000
    return np.array(list_of_featnums),np.array(list_of_results)



mtl_results = collate_for_multiple_settings(1.0, 'RFE')[1][0]
print(mtl_results.shape)
# plt.plot(rfe_results)
# plt.plot(svm_results)
# plt.plot(rfe_lufe_results)
# plt.bar(range(295),rfe_results-svm_results)
plt.bar(range(295),np.sort(mtl_results-rfe_lufe_results))
print(np.mean(mtl_results-rfe_lufe_results))
plt.show()





# plt.plot(list_of_featnums,np.mean(list_of_results,axis=1),label='NN MTL RFE weight 0.0001')
# plt.plot(list_of_featnums,np.mean(list_of_results2,axis=1),label='NN MTL RFE weight 0.001')
# # plt.plot(list_of_featnums,np.mean(list_of_results3,axis=1),lawbel='NN MTL RFE weight 0.01')
# plt.plot(list_of_featnums,np.mean(list_of_results4,axis=1),label='NN MTL mutual info (normalised)', color='red')
# plt.plot(list_of_featnums,np.mean(list_of_results5,axis=1),label='NN MTL RFE (normalised)', color='blue')
# plt.plot(list_of_featnums,np.mean(list_of_results6,axis=1),label='NN MTL ANOVA (normalised)', color='green')
# # plt.plot(list_of_featnums,np.mean(list_of_results6,axis=1),label='NN MTL CHI^2 (normalised)', color='pink')
# plt.plot(list_of_featnums, [np.mean(svm_results)]*len(list_of_featnums), '-.', label='SVM RFE', color='black')
# plt.plot(list_of_featnums, [np.mean(rfe_lufe_results)]*len(list_of_featnums), '-.', label='SVM+ RFE', color='blue')
# plt.plot(list_of_featnums, [np.mean(mi_lufe_results)]*len(list_of_featnums), '-.', label='SVM+ MI', color='red')
# plt.plot(list_of_featnums, [np.mean(anova_lufe_results)]*len(list_of_featnums), '-.', label='SVM+ ANOVA', color='green')
# plt.plot(list_of_featnums, [np.mean(zero_weight_results)]*len(list_of_featnums), '-.', label='Neural net without MTL', color='black')

# ax = plt.subplot()
# ax.set_xscale("log")
# plt.xlabel('Number of unselected features (log scaled)')
# plt.ylabel('Accuracy')
# plt.title('Effect of number of unselected features on MTL neural nets\n classification accuracy  (over 295 datasets)')
# plt.legend(loc='best')
# plt.savefig(get_full_path('Desktop/Privileged_Data/MTLresults/NNResults2'))
# plt.show()


