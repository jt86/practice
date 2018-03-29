from scipy.stats import friedmanchisquare, wilcoxon
from CollateMTLResults2 import collate_mtl_results
import CollateResults
from SingleFoldSlice import Experiment_Setting
from CollateResults import collate_all_datasets
import numpy as np

subset=295
for featsel in ['anova','bahsic','chi2','mi','rfe']:


    s_baseline = np.mean(collate_all_datasets(Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')),axis=1)#[:subset]

    s1 = np.mean(collate_all_datasets(Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')),axis=1)#[:subset]

    print(featsel, wilcoxon(s_baseline, s1)).pvalue #,'!!!' if wilcoxon(s_baseline, s1)>0.001 else None)

    #
    # for lupimethod in ['svmplus', 'dp', 'dsvm']:
    #
    #     s2 = np.mean(collate_all_datasets(Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
    #                                   take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe')),axis=1)#[:subset]
    #
    #     print(featsel, lupimethod, wilcoxon(s1, s2), '!!!' if wilcoxon(s_baseline, s1)>0.001 else None)
    #     print(featsel, lupimethod, wilcoxon(s_baseline, s2), '!!!' if wilcoxon(s_baseline, s1)>0.001 else None)
    #     print(featsel, lupimethod, friedmanchisquare(s_baseline,s1,s2), '!!!' if wilcoxon(s_baseline, s1)>0.001 else None)
    # print('\n')
