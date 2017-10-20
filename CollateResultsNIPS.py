from Get_Full_Path import get_full_path
from SingleFoldSlice import make_directory, Experiment_Setting
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mutual_info_score
from SingleFoldSlice import get_train_and_test_this_fold, get_norm_priv

def collate_single_dataset(s):
    # print(s.name)
    results=np.zeros(10)
    output_directory = get_full_path((
        'Desktop/Privileged_Data/percentinstancesresults/{}/{}{}/').format(s.name,s.dataset, s.datasetnum))
    # print(output_directory)
    assert os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed))),'{} does not exist'.format(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
    with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
        for item in csv.reader(cv_lupi_file):
            results[item[0]]=item[1]
    assert 0 not in results
    if 0 in results:
        # print ("setting = Experiment_Setting(foldnum={}, topk=300, dataset='tech', datasetnum={}, kernel='linear',cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv={}, percentageofinstances=100, take_top_t='top', lupimethod='{}',featsel='{}',classifier='{}')".format(np.where(results==0)[0][0],s.datasetnum,s.percent_of_priv,s.lupimethod,s.featsel,s.classifier))
        print("print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top')"
        .format(np.where(results == 0)[0][0], 300, 'tech', s.datasetnum, 1,  s.lupimethod, s.featsel, s.classifier, s.stepsize, s.percentageofinstances))
    return results




classifier = 'featselector'
lupimethod = 'nolufe'
featsel='rfe'
# for featsel in ['rfe','anova','chi2']:#w,'bahsic']:#,'mi']:#

def collate_diff_percent_instances(datasetnum):
    means_featsel, means_lufe = [],[]
    for instances in [10,20,30,40,50,60,70,80,90]:
        setting_featsel = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod='nolufe',
                 featsel=featsel,classifier='featselector',stepsize=0.1)
        means_featsel.append(np.mean(collate_single_dataset(setting_featsel)))

        setting_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod='svmplus',
                 featsel=featsel,classifier='lufe',stepsize=0.1)
        means_lufe.append(np.mean(collate_single_dataset(setting_lufe)))
    print(means_featsel)
    print(means_lufe)
collate_diff_percent_instances(5)


