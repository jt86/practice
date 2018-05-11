from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all,collate_single_dataset
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path

bbc_lufe_setting = Experiment_Setting(foldnum='all', topk=300, dataset='bbc', datasetnum=0, skfseed=1, kernel='linear',
                        take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='lufe',
                        percent_of_priv=100)


bbc_featsel_setting = Experiment_Setting(foldnum='all', topk=300, dataset='bbc', datasetnum=0, skfseed=1, kernel='linear',
                        take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='featselector',
                        percent_of_priv=100)

for featsel in ['rfe','mi']:
    print(collate_single_dataset(bbc_lufe_setting),collate_single_dataset(bbc_featsel_setting))