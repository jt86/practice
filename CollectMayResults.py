import matplotlib
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats
from os.path import join

may_folder = get_full_path('Desktop/Privileged_Data/MayResults')
skfseed = 0


def get_setting_scores(skf_seed,setting):
    classifier = setting.split('-')[0]
    all_scores = []
    for datasetnum in range(295):
        with open(join(may_folder,setting,'tech{}'.format(datasetnum),'{}-{}.csv'.format(classifier,skfseed))) as result_file:
            dataset_scores = []
            for item in result_file.readlines():
                dataset_scores.append(float(item.split(',')[1].strip('\n')))
        all_scores.append(dataset_scores)
    return np.array(all_scores)


def compare_two_settings(one,two):
    scores_one = np.mean(get_setting_scores(0,one),axis=1)
    scores_two = np.mean(get_setting_scores(0, two),axis=1)
    differences = scores_one-scores_two
    print(len(np.where(differences==0)[0]),len(np.where(differences>0)[0]),len(np.where(differences<0)[0]))
    print(np.mean(differences))
    print(np.mean(scores_one),np.mean(scores_two))


baseline = 'baseline-none-none-300selected-top100priv'
baseline_scores = get_setting_scores(0,baseline)

rfe = 'featselector-nolufe-RFE-300selected-top100priv'
rfe_scores = get_setting_scores(0,rfe)

compare_two_settings(baseline,rfe)

# plt.bar(range(295),compare_two_settings(baseline,rfe))
# plt.show()