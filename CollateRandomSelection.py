from ExperimentSetting import Experiment_Setting
from CollateResults import plot_total_comparison, plot_bars, compare_two_settings
from CollateMTLResults2 import collate_mtl_results, collate_all, plot_bars_for_mtl
# from CollateResults2 import
import numpy as np
import pandas as pd



featsel='rfe'
s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')

s_svm = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')

s_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')


s_random_svm = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='random', classifier='random_featsel_svm')

s_random_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel='random', classifier='random_featsel_svmplus')

names = ['s_baseline', 's_svm', 's_lufe', 's_random_svm', 's_random_lufe']
settings = [s_baseline, s_svm, s_lufe, s_random_svm, s_random_lufe]
for name, setting in zip(names, settings):
    setting = collate_all(setting)
    df = pd.DataFrame(setting).transpose()
    print(df.head())
    df.to_csv('{}.csv'.format(name))
    # with open('{}.csv'.format(name), 'a') as output:
    #     output.save

