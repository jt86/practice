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


    s2 = np.mean(collate_all_datasets(Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')),axis=1)#[:subset]

    print(featsel,wilcoxon(s_baseline, s1))
    print(featsel, wilcoxon(s1, s2))
    print(featsel, wilcoxon(s_baseline, s2))

    print(featsel, friedmanchisquare(s_baseline,s1,s2))

    print('\n')
    #
    # all_results=np.vstack((results1,results2))
    # print(all_results.shape)
    # all_results2= all_results.transpose().tolist()
    # print('list lenght',len(all_results2))
    # print(all_results2[:5])

    # results1=np.random.rand(295)
    # results2=np.random.rand(295)

