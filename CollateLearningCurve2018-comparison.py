from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all, collate_single_dataset
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
import sys
from scipy import stats

means_featsel, means_lufe, means_baseline, means_training_featsel, means_training_lufe, means_training_baseline = [], [], [], [], [], []

pc_range = range(20, 101, 20)
mean_diffs, errors = [], []
#### This part to do for average over all datasets

for percentageofinstances in pc_range:
    s_featsel = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                   kernel='linear',
                                   percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                   lupimethod='nolufe', featsel='mi', classifier='featselector')

    s_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances,
                                take_top_t='top',
                                lupimethod='svmplus', featsel='mi', classifier='lufe')

    # s_feat_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
    #                                   kernel='linear',
    #                                   percent_of_priv=100, percentageofinstances=percentageofinstances,
    #                                   take_top_t='top',
    #                                   lupimethod='nolufe', featsel='mi', classifier='svmtrain')
    #
    # s_lufe_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
    #                                   kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances,
    #                                   take_top_t='top',
    #                                   lupimethod='svmplus', featsel='mi', classifier='lufetrain')
    #
    # s_baseline = Experiment_Setting(foldnum='all', topk='nofeatsel', dataset='tech', datasetnum='all', skfseed=1,
    #                                 kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances,
    #                                 take_top_t='top',
    #                                 lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
    #
    # s_baseline_train = Experiment_Setting(foldnum='all', topk='nofeatsel', dataset='tech', datasetnum='all', skfseed=1,
    #                                       kernel='linear', percent_of_priv=100,
    #                                       percentageofinstances=percentageofinstances,
    #                                       take_top_t='top',
    #                                       lupimethod='nolufe', featsel='nofeatsel', classifier='baselinetrain')
    #
    # means_baseline.append(np.mean(collate_all(s_baseline)))
    # means_training_baseline.append(np.mean(collate_all(s_baseline_train)))
    # means_featsel.append(np.mean(collate_all(s_featsel)))
    # means_lufe.append(np.mean(collate_all(s_lufe)))
    # means_training_featsel.append(np.mean(collate_all(s_feat_train)))
    # means_training_lufe.append(np.mean(collate_all(s_lufe_train)))

    # print()

    print(np.mean(collate_all(s_lufe),axis=1).shape)
    print(np.mean(collate_all(s_featsel), axis=1).shape)

    mean_diff = (np.mean(np.mean(collate_all(s_lufe), axis=1)-np.mean(collate_all(s_featsel), axis=1)))
    error = stats.sem(np.mean(collate_all(s_lufe), axis=1)-np.mean(collate_all(s_featsel), axis=1))
    mean_diffs.append(mean_diff)
    errors.append(error)

fig,ax=plt.subplots()
plt.bar(pc_range,mean_diffs,width=10,yerr=errors)
ax.set_xticks(pc_range)
# plt.set_xticks(pc_range)
ax.set_xlabel('Percentage of training data used')
ax.set_ylabel('Improvement by LUFe over standard feature selection')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/learningcurve-differences.pdf'),format='pdf')
plt.show()