from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
from CombinedPlot import all_plots
# from CollateResults import compare_two_settings, get_graph_labels

topk = 300
lufe_scores = dict()

baseline_score = np.mean(collate_all(Experiment_Setting(foldnum='all', topk='all', dataset='tech',
                        datasetnum='all', skfseed=1, kernel='linear', take_top_t='top', lupimethod='nolufe',
                                                                 featsel='nofeatsel', classifier='baseline')))
# print(baseline_score)

featsels = ['anova','bahsic','chi2','mi','rfe']
for featsel in featsels:
    s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
    s2 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufeshuffle')
    s3 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='luferandom')
    s4 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                            take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')

    lufe_scores[featsel]=([np.mean(collate_all(item))for item in (s1,s2,s3,s4)])
#     print('\n')
#     print([np.mean(collate_all(item))for item in (s1,s2,s3)])
#     # print([np.mean(collate_all(s4) - baseline_score)] * 3)
#     plt.scatter([np.mean(collate_all(s4) - baseline_score)] * 3,[np.mean(collate_all(item))- np.mean(collate_all(s4)) for item in (s1,s2,s3)])
#
# plt.show()

labels_list = ['LUFe', 'LUFe-Shuffle', 'LUFe-Random', 'FeatSel']


##### This part to iterate over feat selection methods and produce LaTeX code for table
num_settings = 4
for i in (featsels):
    print('{} & {:.2f}\% & {:.2f}\% & {:.2f}\% \\\\'.format(i.upper(), *lufe_scores[i][:3]))

#
# num_settings = 4
# bar_width = 1/(num_settings+1)
# indices = np.array(range(len(featsels)))
# fig, ax = plt.subplots(figsize = (7,7))
# for i in range(num_settings):
#     plt.bar(indices + (i*bar_width), [lufe_scores[key][i] for key in featsels], bar_width, label=labels_list[i])
# baseline_line = plt.axhline(y=baseline_score,c='k',linestyle='--',label='ALL')
# # ax.set_xticklabels(['']+featsels)
# plt.xticks(indices + (bar_width*num_settings/2), [i.upper() for i in featsels])
# plt.xlabel('Feature selection metric')
# plt.ylabel('Mean accuracy score (%)')
# plt.ylim((80,87))
# plt.legend()
# plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3a/shufflerandombar'))
# plt.show()


# all_plots(s1,s2,s3,featsel='anova',chapname='chap3a')