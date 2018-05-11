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

s1 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='nolufe',
                       featsel='rfe', classifier='featselector', stepsize=0.1)
svm_results = collate_all(s1)
svm_results = svm_results[:,[-1]].flatten()
indices = (np.argsort(svm_results))


#### get rfe-lufe baseline

s2 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='svmplus',
                       featsel='rfe', classifier='lufe', stepsize=0.1)

rfe_lufe_results = collate_all(s2)
rfe_lufe_results = rfe_lufe_results[:, [-1]].flatten()

#### get mi-lufe baseline

s3 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='svmplus',
                       featsel='mi', classifier='lufe', stepsize=0.1)

mi_lufe_results = collate_all(s3)
mi_lufe_results = mi_lufe_results[:, [-1]].flatten()



#### get mi-lufe baseline

s4 = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='svmplus',
                       featsel='anova', classifier='lufe', stepsize=0.1)

anova_lufe_results = collate_all(s4)
anova_lufe_results = anova_lufe_results[:, [-1]].flatten()


#### load NN results and re-order using
names_list = []
results_dict = {}
results_dict['svm_setting']=np.array(svm_results)[indices]
results_dict['rfe_lufe_results']=np.array(rfe_lufe_results)[indices]


#######

def collate_nn_single_file(filename, featsel, indices=indices):
    nn_results = []
    with open(get_full_path('Desktop/Privileged_Data/MTL_{}_results/{}.csv'.format(featsel,filename)), 'r') as results_file:
        reader = csv.reader(results_file,delimiter=',')
        for count, line in enumerate(reader):
            nn_results.append(float(line[-1]))
    sorted_results =np.array(nn_results)[indices]
    results_dict[filename]=sorted_results
    names_list.append(filename)
    return sorted_results


def collate_for_multiple_settings(weight,featsel):
    list_of_featnums,list_of_results = [],[]
    for num_unsel_feats in [1,2,3]+[item for item in range(10,300,10)]+[item for item in range (1000,2200,100)]+['all']:
        print(num_unsel_feats)
        sorted_results =collate_nn_single_file('MTLresultsfile-3200units-weight{}-numfeats={}-learnrate0.0001'.format(weight,num_unsel_feats),featsel)
        # collate_nn_results('MTLresultsfile-3200units-weight0.001-numfeats={}-learnrate0.0001'.format(num_unsel_feats))
        list_of_featnums.append(num_unsel_feats)
        list_of_results.append(sorted_results)
    list_of_featnums[-1] = 21000
    return np.array(list_of_featnums),np.array(list_of_results)


####################
####################
####################


weight=1
num_unsel_feats=300
featsel = 'RFE'
mtl_300_results =collate_nn_single_file('MTLresultsfile-3200units-weight{}-numfeats={}-learnrate0.0001'.format(weight, num_unsel_feats), featsel)
# plt.plot(range(295),sorted_results)
# plt.plot(range(295),rfe_lufe_results[indices])
# print(sorted_results)

# plt.plot(range(295),sorted_results-rfe_lufe_results)
plt.show()
improvements_list = mtl_300_results - rfe_lufe_results
print(len(np.where(improvements_list>0)[0]))
print(len(np.where(improvements_list<0)[0]))
print(len(np.where(improvements_list==0)[0]))
# plt.bar(range(295),improvements_list)
plt.bar(range(295),mtl_300_results, color='blue')
plt.bar(np.array(range(295))+0.5,rfe_lufe_results, color='red')

ax = plt.subplot(111)
ax.bar(x-0.2, y,width=0.2,color='b',align='center')
ax.bar(x, z,width=0.2,color='g',align='center')
ax.bar(x+0.2, k,width=0.2,color='r',align='center')
ax.xaxis_date()


plt.show()
sys.exit()


####################
####################
####################

# list_of_featnums,list_of_results =collate_for_multiple_settings(weight=0.0001,featsel='RFE-notnormed')
# list_of_featnums2,list_of_results2 =collate_for_multiple_settings(weight=0.001,featsel='RFE-notnormed')
# list_of_featnums3,list_of_results3 =collate_for_multiple_settings(weight=0.01,featsel='RFE-notnormed')
list_of_featnums4,list_of_results4 =collate_for_multiple_settings(weight=1,featsel='MI')
list_of_featnums5,list_of_results5 =collate_for_multiple_settings(weight=1,featsel='RFE')
list_of_featnums6,list_of_results6 =collate_for_multiple_settings(weight=1,featsel='ANOVA')
# list_of_featnums7,list_of_results7 =collate_for_multiple_settings(weight=1,featsel='CHI2')

list_of_featnums=list_of_featnums5
print(list_of_featnums)
#
# def compare_nn_svm(nn_setting,svm_setting='svm_setting'):
#     svm_results= results_dict[svm_setting]
#     nn_results = results_dict[nn_setting]
#     svm_improvements=np.array((svm_results-nn_results))
#     print('{}: svm better: {}, nn better: {}, mean: {}'.format(nn_setting,
#                                                                             len(np.where(svm_improvements > 0)[0]),
#                                                                             (len(np.where(svm_improvements < 0)[0])),np.mean(results_dict[nn_setting])))
#     return np.mean(results_dict[nn_setting])



list_of_scores = []
# for item in names_list:
#     np.set_printoptions(precision=3)
#     # print('\n {} units: mean accuracy = {}, std = {}'.format(name,np.mean(item),np.std(item)))
#     list_of_scores.append(compare_nn_svm(item))




print(len(svm_results))
print(np.mean(svm_results))


zero_weight_results = np.mean(collate_nn_single_file('MTLresultsfile-3200units-weight0-numfeats=300-learnrate0.0001',featsel='RFE-notnormed'))


# plt.plot(list_of_featnums,np.mean(list_of_results,axis=1),label='NN MTL RFE weight 0.0001')
# plt.plot(list_of_featnums,np.mean(list_of_results2,axis=1),label='NN MTL RFE weight 0.001')
# plt.plot(list_of_featnums,np.mean(list_of_results3,axis=1),lawbel='NN MTL RFE weight 0.01')
plt.plot(list_of_featnums,np.mean(list_of_results4,axis=1),label='NN MTL mutual info (normalised)', color='red')
plt.plot(list_of_featnums,np.mean(list_of_results5,axis=1),label='NN MTL RFE (normalised)', color='blue')
plt.plot(list_of_featnums,np.mean(list_of_results6,axis=1),label='NN MTL ANOVA (normalised)', color='green')
# plt.plot(list_of_featnums,np.mean(list_of_results6,axis=1),label='NN MTL CHI^2 (normalised)', color='pink')
plt.plot(list_of_featnums, [np.mean(svm_results)]*len(list_of_featnums), '-.', label='SVM RFE', color='black')
plt.plot(list_of_featnums, [np.mean(rfe_lufe_results)]*len(list_of_featnums), '-.', label='SVM+ RFE', color='blue')
plt.plot(list_of_featnums, [np.mean(mi_lufe_results)]*len(list_of_featnums), '-.', label='SVM+ MI', color='red')
plt.plot(list_of_featnums, [np.mean(anova_lufe_results)]*len(list_of_featnums), '-.', label='SVM+ ANOVA', color='green')
plt.plot(list_of_featnums, [np.mean(zero_weight_results)]*len(list_of_featnums), '-.', label='Neural net without MTL', color='black')

ax = plt.subplot()
ax.set_xscale("log")
plt.xlabel('Number of unselected features (log scaled)')
plt.ylabel('Accuracy')
plt.title('Effect of number of unselected features on MTL neural nets\n classification accuracy  (over 295 datasets)')
plt.legend(loc='best')
plt.savefig(get_full_path('Desktop/Privileged_Data/MTLresults/NNResults2'))
plt.show()


