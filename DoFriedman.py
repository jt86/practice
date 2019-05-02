'''
Module for statistical significance testing of results
'''
from scipy.stats import friedmanchisquare, wilcoxon
import CollateResults
from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from CollateMTLResults2 import collate_mtl_results
subset=295

featsel='rfe'




s_baseline = np.mean(collate_all(Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                            take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')), axis=1)#[:subset]


for featsel in ['anova','bahsic','chi2','mi','rfe']:
    print(featsel+'----------------------------------------------')
    #
    # s1 = np.mean(collate_all(Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
    #                                             take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')), axis=1)#[:subset]
    #
    # print('featsel vs baseline featsel = {}  score = {}'.format(featsel, wilcoxon(s_baseline, s1)),'!!!' if wilcoxon(s_baseline, s1).pvalue>0.001 else None)

    #
    for lupimethod in ['svmplus']:#, 'dp', 'dsvm']:
        print('\n lupimethod ={} -------------------------'.format(lupimethod))
        s2 = np.mean(collate_all(Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe')),axis=1)#[:subset]

        # print('lufe vs featsel',featsel, lupimethod, wilcoxon(s1, s2), '!!!' if wilcoxon(s1, s2).pvalue>0.001 else None)
        # print('lufe vs all',featsel, lupimethod, wilcoxon(s_baseline, s2), '!!!' if wilcoxon(s_baseline, s2).pvalue>0.001 else None)
        # print(featsel, lupimethod, friedmanchisquare(s_baseline,s1,s2), '!!!' if wilcoxon(s_baseline, s1).pvalue>0.001 else None)

        mtl_results = (np.mean(collate_mtl_results(featsel.upper(), 300), axis=1))
        print(wilcoxon(s2, mtl_results))

        print(np.mean(mtl_results-s2),np.mean(s2),np.mean(mtl_results))

        # print(mtl_results)
        # print(s2)

    print('\n')
