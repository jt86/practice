import numpy as np
from CollateResults import plot_bars, collate_all_datasets,get_lufe_improvements_per_fold
from ExperimentSetting import Experiment_Setting

s1 = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')


for featsel in ['rfe','bahsic','anova','chi2','mi']:
    print('\n')
    for lupimethod in ['svmplus','dsvm','dp']:
        s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                     take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='svmreverse')
        s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                     take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe')
        print(featsel,'\t', lupimethod,'\t', round(np.mean(collate_all_datasets(s)),3),'\t',round(np.mean(collate_all_datasets(s2)),3),'\t',round(np.mean(collate_all_datasets(s2))-np.mean(collate_all_datasets(s))*10,3))

# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for lupimethod in ['svmplus','dsvm','dp']:
#         s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                      take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufereverse')
#         s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                      take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe')
#         print(featsel,'\t', lupimethod,'\t', np.mean(collate_all_datasets(s)),np.mean(collate_all_datasets(s2)),np.mean(collate_all_datasets(s2))-np.mean(collate_all_datasets(s))*10)




#####  PER FOLD
# for featsel in ['rfe', 'bahsic', 'anova', 'chi2', 'mi']:
#     svm = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='nolufe', featsel=featsel,
#                              classifier='featselector')
#     lufe = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='svmplus', featsel=featsel,
#                               classifier='lufe')
#     get_lufe_improvements_per_fold(svm, lufe)

##### PLOTTING

# Plotting feature selection VS baseline
# for featsel in ['rfe','mi','anova','chi2','bahsic']:
#     s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
#     plot_bars(s1, s2)
#
# # Plotting LUFe VS baseline
# for featsel in ['rfe','mi','anova','chi2','bahsic']:
#     s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
#     plot_bars(s1, s2)
#
# # Plotting LUFe VS feature selection
# for featsel in ['rfe','mi','anova','chi2','bahsic']:
#     s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
#     s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
#     plot_bars(s1, s2)
#
#