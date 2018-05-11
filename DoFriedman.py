from scipy.stats import friedmanchisquare, wilcoxon
from CollateMTLResults2 import collate_mtl_results
import CollateResults
from SingleFoldSlice import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from CollateMTLResults2 import collate_mtl_results
subset=295

featsel='rfe'
mtl_results = (np.mean(collate_mtl_results(featsel.upper(), 300),axis=1))
print(mtl_results.shape)
# for featsel in ['anova','bahsic','chi2','mi','rfe']:
for featsel in ['rfe']:


    s2 = np.mean(collate_all(Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='rbf',
                                                take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')), axis=1)#[:subset]

    print(featsel, wilcoxon(mtl_results, s2))#,'!!!' if wilcoxon(s_baseline, s1)>0.001 else None)

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
