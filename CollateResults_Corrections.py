from ExperimentSetting import Experiment_Setting
from CollateResults import plot_total_comparison, plot_bars, compare_two_settings
from CollateMTLResults2 import collate_mtl_results, collate_all, plot_bars_for_mtl
# from CollateResults2 import
import numpy as np
import pandas as pd


featsel='rfe'
s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')



results_table = pd.DataFrame(index=['name']+['ds{}'.format(i) for i in range(295)])
print(results_table)

def make_column(setting):
    mean_results = np.mean(collate_all(setting), axis=1)
    col_title = [setting.name]
    return col_title, mean_results

for setting in [s_baseline, s1, s2]:
    name, results = make_column(setting)
    results_table[name] = results
