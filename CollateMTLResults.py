import os
from Get_Full_Path import get_full_path
import csv
import pandas
import numpy as np
from CollateResults import collate_single_dataset,collate_all_datasets
from SingleFoldSlice import Experiment_Setting
from matplotlib import pyplot as plt

#
# def collate_nn_results(filename):
#     nn_results = []
#     with open(get_full_path('Desktop/Privileged_Data/{}'.format(filename)), 'r') as results_file:
#         reader = csv.reader(results_file,delimiter=',')
#         for count, line in enumerate(reader):
#             nn_results.append(float(line[-1]))if line[5]=='0' else None
#         print(len(nn_results))
#     return np.array(nn_results)


#### get svm baseline

s = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='nolufe',
                       featsel='rfe', classifier='featselector', stepsize=0.1)
svm_results = collate_all_datasets(s)
svm_results = svm_results[:,[-1]].flatten()
indices = (np.argsort(svm_results))


#### load NN results and re-order using
names_list = []
results_dict = {}
results_dict['svm_setting']=np.array(svm_results)[indices]

def collate_nn_results(filename, indices=indices):
    nn_results = []
    with open(get_full_path('Desktop/Privileged_Data/{}'.format(filename)), 'r') as results_file:
        reader = csv.reader(results_file,delimiter=',')
        for count, line in enumerate(reader):
            nn_results.append(float(line[-1]))

    results_dict[filename]=np.array(nn_results)[indices]
    names_list.append(filename)


nn_results3200_weight0 = collate_nn_results('MTLresultsfile-3200units-weight0.csv')
nn_results6400_weight0 = collate_nn_results('MTLresultsfile-6400units-weight0.csv')
nn_results3200_weight0_smallstep = collate_nn_results('MTLresultsfile-3200units-weight0-learningrate10-5.csv')

nn_results320_weight0 = collate_nn_results('MTLresultsfile-320units-weight0.csv')
nn_results320_weight0_01 = collate_nn_results('MTLresultsfile-320units-weight0.01.csv')
nn_results320_weight1 = collate_nn_results('MTLresultsfile-320units-weight1.csv')
nn_results320_weight100 = collate_nn_results('MTLresultsfile-320units-weight100.csv')



def compare_nn_svm(nn_setting,svm_setting='svm_setting'):
    svm_results= results_dict[svm_setting]
    nn_results = results_dict[nn_setting]
    svm_improvements=np.array((svm_results-nn_results))
    print('{}: svm better: {}, nn better: {}, equal: {}, total: 295'.format(nn_setting,len(np.where(svm_improvements>0)[0]),
    (len(np.where(svm_improvements<0)[0])),
    (len(np.where(svm_improvements==0)[0]))))
   #  plt.plot(range(295),svm_results)
   #  plt.plot(range(295),nn_results6400)
   #  plt.show()

for item in names_list:
    np.set_printoptions(precision=3)
    # print('\n {} units: mean accuracy = {}, std = {}'.format(name,np.mean(item),np.std(item)))
    compare_nn_svm(item)
