from Get_Full_Path import get_full_path
from SingleFoldSlice import make_directory, Experiment_Setting
import os
import csv
import numpy as np

def collate_single_dataset(s):
    # print(s.name)
    results=np.zeros(10)
    output_directory = get_full_path((
        'Desktop/Privileged_Data/JuneResults/{}/{}{}/').format(s.name,s.dataset, s.datasetnum))

    assert os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
    with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
        for item in csv.reader(cv_lupi_file):
            results[item[0]]=item[1]
    assert 0 not in results
    # if 0 in results:
    #     print ("setting = Experiment_Setting(foldnum={}, topk=300, dataset='tech', datasetnum={}, kernel='linear',cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod={},featsel={},classifier={})".format(s.foldnum,s.datasetnum,s.lupimethod,s.featsel,s.classifier))
    #     print(s.datasetnum)
    return results



def collate_all_datasets(s,num_datasets=295):
    all_results = []
    for datasetnum in range(num_datasets):
        s.datasetnum=datasetnum
        all_results.append(collate_single_dataset(s))
    # print(s.name,1-np.mean(all_results))
    return (np.array(all_results))


def compare_two_lists(setting_one, setting_two, name_one, name_two):
    improvements_list=np.zeros(295)
    setting_one = np.mean(setting_one, axis=1)
    setting_two = np.mean(setting_two, axis=1)
    print(setting_one.shape, setting_two.shape)
    for count,(score_one, score_two) in enumerate(zip(setting_one, setting_two)):
        improvements_list[count]= (score_one - score_two) # this value is positive if score one is better
    print('{} better: {}; {} better: {}; equal: {}; mean improvement={}%'.format(name_one,len(np.where(improvements_list > 0)[0]),
          name_two,len(np.where(improvements_list < 0)[0]),len(np.where(improvements_list==0)[0]),np.mean(improvements_list)))
    return(improvements_list)

def compare_two_settings(s1,s2):
    compare_two_lists(collate_all_datasets(s1),collate_all_datasets(s2),s1.name,s2.name)

setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
collate_all_datasets(setting)

setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='lufe')
print(np.mean(collate_all_datasets(setting)))

for featsel in ['rfe','anova','chi2','mi','bahsic']:
    s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                 take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    print(s.name,np.mean(collate_all_datasets(s)))

print('\n')
# Add results for lufe settings to dataframe
for lufe in ['dp', 'svmplus','dsvm']:
    for featsel in ['rfe','anova','chi2','mi','bahsic']:
        s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                     take_top_t='top', lupimethod=lufe, featsel=featsel, classifier='lufe')
        print(s.name, np.mean(collate_all_datasets(s)))


#
# setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                              take_top_t='top', lupimethod='nolufe', featsel='bahsic', classifier='featselector')
# collate_all_datasets(setting)

#

