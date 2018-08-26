from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all, collate_single_dataset
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
import sys

means_featsel, means_lufe, means_baseline, means_training_featsel, means_training_lufe, means_training_baseline = [], [], [], [], [], []

pc_range=range(20,101,20)

#### This part to do for average over all datasets

for percentageofinstances in pc_range:
    s_featsel = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                   percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                   lupimethod='nolufe', featsel='mi', classifier='featselector')

    s_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                lupimethod='svmplus', featsel='mi', classifier='lufe')

    s_feat_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                   percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                   lupimethod='nolufe', featsel='mi', classifier='svmtrain')

    s_lufe_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                lupimethod='svmplus', featsel='mi', classifier='lufetrain')

    s_baseline = Experiment_Setting(foldnum='all', topk='nofeatsel', dataset='tech', datasetnum='all', skfseed=1,
                                    kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances,
                                    take_top_t='top',
                                    lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')

    s_baseline_train = Experiment_Setting(foldnum='all', topk='nofeatsel', dataset='tech', datasetnum='all', skfseed=1,
                                    kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances,
                                    take_top_t='top',
                                    lupimethod='nolufe', featsel='nofeatsel', classifier='baselinetrain')

    means_baseline.append(np.mean(collate_all(s_baseline)))
    means_training_baseline.append(np.mean(collate_all(s_baseline_train)))
    means_featsel.append(np.mean(collate_all(s_featsel)))
    means_lufe.append(np.mean(collate_all(s_lufe)))
    means_training_featsel.append(np.mean(collate_all(s_feat_train)))
    means_training_lufe.append(np.mean(collate_all(s_lufe_train)))

    print()



print(np.mean(collate_all(s_baseline).shape))
# sys.exit()
#### This part to do for individual datasets

# for datasetnum in range(10):
#     means_featsel, means_lufe, means_training_featsel, means_training_lufe = [], [], [], []
#     for percentageofinstances in pc_range:
#         s_featsel = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1,
#                                        kernel='linear',
#                                        percent_of_priv=100, percentageofinstances=percentageofinstances,
#                                        take_top_t='top',
#                                        lupimethod='nolufe', featsel='mi', classifier='featselector')
#
#         s_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1,
#                                     kernel='linear', percent_of_priv=100,
#                                     percentageofinstances=percentageofinstances, take_top_t='top',
#                                     lupimethod='svmplus', featsel='mi', classifier='lufe')
#
#         s_feat_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1,
#                                           kernel='linear',
#                                           percent_of_priv=100, percentageofinstances=percentageofinstances,
#                                           take_top_t='top',
#                                           lupimethod='nolufe', featsel='mi', classifier='svmtrain')
#
#         s_lufe_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1,
#                                           kernel='linear', percent_of_priv=100,
#                                           percentageofinstances=percentageofinstances, take_top_t='top',
#                                           lupimethod='svmplus', featsel='mi', classifier='lufetrain')
#         means_featsel.append(np.mean(collate_single_dataset(s_featsel)))
#         means_lufe.append(np.mean(collate_single_dataset(s_lufe)))
#         means_training_featsel.append(np.mean(collate_single_dataset(s_feat_train)))
#         means_training_lufe.append(np.mean(collate_single_dataset(s_lufe_train)))

    ### This part to produce the learning curve plot

plt.plot(pc_range,[100-item for item in means_baseline], label='ALL-SVM+ on test set', color='k')
plt.plot(pc_range,[100-item for item in means_training_baseline], label='ALL-SVM+ on train set', color='k',linestyle=':')
plt.plot(pc_range,[100-item for item in means_featsel], label='FeatSel-MI-SVM on test set', color='b')
plt.plot(pc_range,[100-item for item in means_training_featsel], label='FeatSel-MI-SVM on train set', color='b', linestyle=':')
plt.plot(pc_range,[100-item for item in means_lufe], label='LUFe-MI-SVM+ on test set', color='r')
plt.plot(pc_range,[100-item for item in means_training_lufe], label='LUFe-MI-SVM+ on train set', color='r',linestyle=':')


plt.legend(loc='best')
plt.xlabel('Percent of training data used')
plt.ylabel('Mean error (%)')
# plt.ylim(0,30)
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3c/learningcurve-dataset_new.pdf'),format='pdf')
# plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/learningcurve-dataset{}.pdf'.format(datasetnum)),format='pdf')
# plt.show()

plt.clf()




### This part to produce the table

for item in zip(pc_range,means_baseline, means_featsel, means_lufe, means_training_baseline, means_training_featsel, means_training_lufe):
    print(('\% & ').join(str(round(i,1)) for i in item)+'\% \\\\')
    # print((' & ').join(str(i) for i in item))

print('train data')
for i,j in zip(means_training_lufe, means_training_featsel):
    print(i,j, j-i)
print('test data')
for i,j in zip(means_lufe, means_featsel):
    print(i,j, j-i)