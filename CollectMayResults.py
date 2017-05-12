import matplotlib
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats
from os.path import join

may_folder = get_full_path('Desktop/Privileged_Data/MayResults')
# mid_folder = get_full_path('Desktop/Privileged_Data/TestingCodeResultsMIDPARAM')

def get_setting_scores(skfseed,setting, folder):
    print(setting)
    classifier = setting.split('-')[0]
    all_scores = []
    for datasetnum in range(295):
        with open(join(folder,setting,'tech{}'.format(datasetnum),'{}-{}.csv'.format(classifier,skfseed))) as result_file:
            dataset_scores = []
            for item in result_file.readlines():
                dataset_scores.append(float(item.split(',')[1].strip('\n')))
        all_scores.append(np.array(dataset_scores))
    all_scores= (np.array(all_scores))
    mean_score = np.mean(all_scores)
    for index,item in enumerate(all_scores):
        if len(item) !=10: print (index)
    print('setting = {}, seed ={}, scores = {}'.format(setting, skfseed, np.mean(all_scores)))
    return np.array(all_scores)


def compare_two_settings(one,two):
    scores_one = np.mean(get_setting_scores(1,one, old_norm_folder),axis=1)
    scores_two = np.mean(get_setting_scores(1, two,old_norm_folder),axis=1)
    differences = scores_one-scores_two
    print(''.format(len(np.where(differences==0)[0]),len(np.where(differences>0)[0]),len(np.where(differences<0)[0])))
    print(np.mean(differences))
    print((1-(np.mean(scores_one)))*100,(1-(np.mean(scores_two)))*100)


baseline = 'baseline-none-none-300selected-top100priv'
# get_setting_scores(0,baseline, may_folder)

baseline = 'baseline-nolufe-nofeatsel-allselected-top100priv'
# get_setting_scores(1,baseline, may_folder)

rfe = 'featselector-nolufe-RFE-300selected-top100priv'
# get_setting_scores(0,rfe, may_folder)
# get_setting_scores(1,rfe, may_folder)

lufe = 'lufe-svmplus-RFE-300selected-top100priv'
# get_setting_scores(0,lufe, may_folder)
# get_setting_scores(1,lufe, may_folder)

print('\n ----old normalisation')

old_norm_folder = get_full_path('Desktop/Privileged_Data/OldNormalisation')
baseline_scores = get_setting_scores(1,baseline,old_norm_folder)
rfe_scores = get_setting_scores(1,rfe,old_norm_folder)
lufe_scores = get_setting_scores(1,lufe,old_norm_folder)

compare_two_settings(baseline,lufe)
#
# plt.bar(range(295),compare_two_settings(baseline,rfe))
# plt.show()

from ProcessNumpyResults import get_errors, all_baseline, lufe_baseline, svm_baseline
print (get_errors(all_baseline).shape)
print (100-np.mean(get_errors(all_baseline)),(100-np.mean(get_errors(lufe_baseline))),100-np.mean(get_errors(svm_baseline)))

