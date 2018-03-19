from Get_Full_Path import get_full_path
import os
from ExperimentSetting import Experiment_Setting
import csv
import numpy as np

classifier = 'featselector'
lupimethod = 'nolufe'
featsel='rfe'



def collate_single_dataset_single_fold(s):
    output_directory = get_full_path(('Desktop/Privileged_Data/AllResults/{}/{}/{}{}/').format(s.kernel,s.name,s.dataset, s.datasetnum))
    assert os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed))),'{} does not exist'.format(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
    with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
        for item in csv.reader(cv_lupi_file):
            if int(item[0])==s.foldnum:
                return item[1]

def collate_all_datasets_single_fold(s,num_datasets=295):
    all_results = []
    for datasetnum in range(num_datasets):
        s.datasetnum=datasetnum
        all_results.append(collate_single_dataset_single_fold(s))
    # print(s.name,1-np.mean(all_results))
    return (np.array(all_results,dtype=float))


rbf_results = collate_all_datasets_single_fold(s=Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum='all', kernel='rbf',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='svmplus',
                       featsel='rfe', classifier='lufe', stepsize=0.1))

lin_results = collate_all_datasets_single_fold(s=Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum='all', kernel='linear',
                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                       take_top_t='top', lupimethod='svmplus',
                       featsel='rfe', classifier='lufe', stepsize=0.1))

# print(rbf_results-lin_results)
plt.bar(range(295),rbf_results-lin_results)
plt.show()
print(np.mean(rbf_results-lin_results))