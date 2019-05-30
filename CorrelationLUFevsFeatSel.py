from ExperimentSetting import Experiment_Setting
from CollateResults import plot_total_comparison, plot_bars, compare_two_settings
from CollateMTLResults2 import collate_mtl_results, collate_all, plot_bars_for_mtl
# from CollateResults2 import
import numpy as np
import pandas as pd

############ FOR CHAPTER 1

featsel='rfe'
############ Top 300

s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')